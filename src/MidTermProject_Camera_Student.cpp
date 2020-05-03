#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

    /**************************************/
    /* INIT VARIABLES AND DATA STRUCTURES */
    /**************************************/

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer(dataBufferSize); // sequence of data frames which are held in memory at the same time
    int pos_in_buffer = -1; // Position in the circular buffer, will be increased to 0 at the first iteration
    bool bVis = true;            // visualize results

    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // Loop over all detectors
    vector<string> detectorTypes{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    for (auto const &detectorType: detectorTypes) {
        // Loop over all descriptors
        vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
        for (auto const &descriptorType: descriptorTypes) {

            // The Akaze detector only works with the Akaze descriptor
            if (detectorType == "AKAZE" && descriptorType != "AKAZE")
                continue;
            if (descriptorType == "AKAZE" && detectorType != "AKAZE")
                continue;
            // The SIFT detector doesn't work with the ORB descriptor
            if (detectorType == "SIFT" && descriptorType == "ORB")
                continue;
            // The SIFT descriptor does not work with BF matcher
            if (descriptorType == "SIFT" && matcherType == "MAT_BF")
                continue;

            string descriptorTypeString = (descriptorType == "SIFT") ? "DES_HOG" : "DES_BINARY";

            // Loop over all images
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                DataFrame frame;
                frame.cameraImg = imgGray;
                pos_in_buffer = (pos_in_buffer + 1) % dataBufferSize;
                // Beware, remainder of division can be negative in C++
                auto prev_pos_in_buffer = std::abs((pos_in_buffer - 1)) % dataBufferSize;
                dataBuffer[pos_in_buffer] = frame;

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /**************************/
                /* DETECT IMAGE KEYPOINTS */
                /**************************/

                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType == "SHITOMASI")
                    detGoodFeaturesToTrack(keypoints, imgGray, bVis);
                else if (detectorType == "HARRIS")
                    detKeypointsHarris(keypoints, imgGray, bVis);
                else if (detectorType == "FAST")
                    detKeypointsModern(keypoints, img, "FAST", bVis);
                else if (detectorType == "BRISK")
                    detKeypointsModern(keypoints, img, "BRISK", bVis);
                else if (detectorType == "ORB")
                    detKeypointsModern(keypoints, img, "ORB", bVis);
                else if (detectorType == "AKAZE")
                    detKeypointsModern(keypoints, img, "AKAZE", bVis);
                else if (detectorType == "SIFT")
                    detKeypointsModern(keypoints, img, "SIFT", bVis);
                else {
                    cout << "Unknown detector type: " << detectorType << endl;
                    exit(-1);
                }

                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // Only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle) {
                    vector<cv::KeyPoint> cropped_keypoints;
                    for (const auto &keypoint: keypoints)
                        if (keypoint.pt.x >= vehicleRect.x && keypoint.pt.y >= vehicleRect.y &&
                            keypoint.pt.x < vehicleRect.x + vehicleRect.width &&
                            keypoint.pt.y < vehicleRect.y + vehicleRect.height)
                            cropped_keypoints.emplace_back(keypoint);
                    keypoints = cropped_keypoints;
                }

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = true; // TODO remember to remove before collecting final output
                if (bLimitKpts) {
                    int maxKeypoints = 50;

                    // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    if (detectorType == "SHITOMASI" || detectorType == "HARRIS")
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end()); // TODO is this efficient?
                    else
                        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // Store keypoints for current frame in the data buffer
                dataBuffer[pos_in_buffer].keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /********************************/
                /* EXTRACT KEYPOINT DESCRIPTORS */
                /********************************/

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                // string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints(dataBuffer[pos_in_buffer].keypoints,
                              dataBuffer[pos_in_buffer].cameraImg,
                              descriptors,
                              descriptorType);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                dataBuffer[pos_in_buffer].descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (imgIndex > 0) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors(dataBuffer[prev_pos_in_buffer].keypoints, dataBuffer[pos_in_buffer].keypoints,
                                     dataBuffer[prev_pos_in_buffer].descriptors, dataBuffer[pos_in_buffer].descriptors,
                                     matches, descriptorTypeString, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    dataBuffer[pos_in_buffer].kptMatches = matches;

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    if (bVis) {
                        cv::Mat matchImg = (dataBuffer[pos_in_buffer].cameraImg).clone();
                        cv::drawMatches(dataBuffer[prev_pos_in_buffer].cameraImg,
                                        dataBuffer[prev_pos_in_buffer].keypoints,
                                        dataBuffer[pos_in_buffer].cameraImg, dataBuffer[pos_in_buffer].keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName =
                                "Matching keypoints with " + detectorType + " " + descriptorType + " " + matcherType;
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        // cv::waitKey(0); // wait for key to be pressed
                    }
                } // Keypoints matching
            } // Loop over images
        } // Loop over all descriptor types
    } // Loop over all detector types
    return 0;
}