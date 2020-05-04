#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

class Timer {
    double startTime = static_cast<double>(cv::getTickCount());
public:
    double elapsed() const {
        double res = (static_cast<double >(cv::getTickCount()) - startTime) / cv::getTickFrequency();
        return res;
    }

    void reset() {
        startTime = static_cast<double>(cv::getTickCount());
    }

    double fetch_and_restart() {
        auto savedStart = startTime;
        reset();
        double res = (static_cast<double >(cv::getTickCount()) - savedStart) / cv::getTickFrequency();
        return res;
    }
};


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
    const char dir_separator='/';
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer(dataBufferSize); // sequence of data frames which are held in memory at the same time
    int pos_in_buffer = -1; // Position in the circular buffer, will be increased to 0 at the first iteration
    bool bVis = true;            // visualize results

    vector<string> detectorTypes{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    // ---Statistics to be collected and dumped in files
    // Number of keypoints on the preceding vehicle, by image and by detector
    map<pair<string, string>, int> keypoints_count;
    // Average and sample variance of the neighborhood size, by image and by detector
    map<pair<string, string>, pair<float, float>> neighborhood_size;
    // Total number of matched keypoints (summed across all images), by detector type and by descriptor type
    map<pair<string, string>, int> matched_count;
    // Total time (summed across all images) for keypoints detection plus descriptors extraction, by detector type and
    // descriptor type. TODO measurement unit?
    map<pair<string, string>, float> detect_extract_time;
    set<string> allFileNames; // Used to save reports

    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // Loop over all detectors
    for (auto const &detectorType: detectorTypes) {
        // Loop over all descriptors
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
                string fileName = imgPrefix + imgNumber.str() + imgFileType;
                fileName = fileName.substr(fileName.find_last_of(dir_separator) + 1);
                allFileNames.insert(fileName);
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

                Timer theTimer;

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

                auto elapsed = theTimer.elapsed();

                auto detect_descr = make_pair(detectorType, descriptorType);
                if (detect_extract_time.find(detect_descr) == detect_extract_time.end())
                    detect_extract_time[detect_descr] = elapsed;
                else
                    detect_extract_time[detect_descr] += elapsed; // TODO check this works as intended!

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
                bool bLimitKpts = false; // TODO remember to remove before collecting final output
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

                // Compute average and sample variance of the keypoints neighborhood size
                double sum = 0;
                for_each(keypoints.begin(), keypoints.end(), [&sum](cv::KeyPoint kp) { sum += kp.size; });
                double avg = sum / keypoints.size();

                double sq_dev = 0;
                for_each(keypoints.begin(), keypoints.end(), [avg, &sq_dev](cv::KeyPoint kp) {
                    auto dev = kp.size - avg;
                    sq_dev += dev * dev;
                });
                double sample_var = sq_dev / (keypoints.size() - 1.);

                cout << "Average = " << avg << " Sample std dev. = " << std::sqrt(sample_var) << endl;

                // Fill in statistics
                keypoints_count[make_pair(fileName, detectorType)] = keypoints.size();
                neighborhood_size[make_pair(fileName, detectorType)] = make_pair(avg, sample_var);

                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /********************************/
                /* EXTRACT KEYPOINT DESCRIPTORS */
                /********************************/

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                theTimer.reset();
                cv::Mat descriptors;
                // string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints(dataBuffer[pos_in_buffer].keypoints,
                              dataBuffer[pos_in_buffer].cameraImg,
                              descriptors,
                              descriptorType);

                //// EOF STUDENT ASSIGNMENT

                // store descriptors for current frame to end of data buffer
                dataBuffer[pos_in_buffer].descriptors = descriptors;

                elapsed = theTimer.elapsed();
                detect_extract_time[detect_descr] += elapsed; // TODO check this works as intended!


                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (imgIndex > 0) // wait until at least two images have been processed
                {

                    /******************************/
                    /* MATCH KEYPOINT DESCRIPTORS */
                    /******************************/

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

                    // auto detect_descr = make_pair(detectorType, descriptorType);
                    if (matched_count.find(detect_descr) == matched_count.end())
                        matched_count[detect_descr] = matches.size();
                    else
                        matched_count[detect_descr] += matches.size(); // TODO check this actually works as intended!

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
    // Save statistics in files

    // Number of keypoints on the preceding vehicle, by image and by detector
    //map<pair<string, string>, int> keypoints_count;

    // Average and sample variance of the neighborhood size, by image and by detector
    //map<pair<string, string>, pair<float, float>> neighborhood_size;

    // Total number of matched keypoints (summed across all images), by detector type and by descriptor type
    //map<pair<string, string>, int> matched_count;

    // Total time (summed across all images) for keypoints detection plus descriptors extraction, by detector type and
    // descriptor type. TODO measurement unit?
    //map<pair<string, string>, float> detect_extract_time;

    ofstream image_detect_file(dataPath + "stats.txt");
    if (!image_detect_file.is_open()) {
        cout << "Unable to open file for writing" << endl;
        exit(-1);
    }

    auto write_detectors_header = [&image_detect_file, &detectorTypes]() {
        for (const auto &detector_name: detectorTypes) {
            image_detect_file << detector_name << " ";
        }
        image_detect_file << endl;
    };

    image_detect_file << "Number_of_keypoints_on_the_preceding_vehicle,_by_image_and_by_detector" << endl;
    write_detectors_header();
    for (const auto &file_name: allFileNames) {
        image_detect_file << file_name << " ";
        for (const auto &detector_name: detectorTypes) {
            image_detect_file << keypoints_count[make_pair(file_name, detector_name)] << " ";
        }
        image_detect_file << endl;
    }

    image_detect_file << "Average_of_the_neighborhood_size,_by_image_and_by_detector" << endl;
    write_detectors_header();
    for (const auto &file_name: allFileNames) {
        image_detect_file << file_name << " ";
        for (const auto &detector_name: detectorTypes) {
            image_detect_file << neighborhood_size[make_pair(file_name, detector_name)].first << " ";
        }
        image_detect_file << endl;
    }

    image_detect_file << "Sample_std_dev_of_the_neighborhood_size,_by_image_and_by_detector" << endl;
    write_detectors_header();
    for (const auto &file_name: allFileNames) {
        image_detect_file << file_name << " ";
        for (const auto &detector_name: detectorTypes)
            image_detect_file << std::sqrt(neighborhood_size[make_pair(file_name, detector_name)].second) << " ";
        image_detect_file << endl;
    }

    auto write_descriptors_header = [&image_detect_file, &descriptorTypes]() {
        for (const auto &descriptor_name: descriptorTypes)
            image_detect_file << descriptor_name << " ";
        image_detect_file << endl;
    };

    image_detect_file
            << "Total_number_of_matched_keypoints_(summed_across_all_images),_by_detector_type_and_by_descriptor_type"
            << endl;
    write_descriptors_header();
    for (const auto &detector_name: detectorTypes) {
        image_detect_file << detector_name << " ";
        for (const auto &descriptor_name: descriptorTypes)
            image_detect_file << matched_count[make_pair(detector_name, descriptor_name)] << " ";
        image_detect_file << endl;
    }

    image_detect_file
            << "Total_time_(summed_across_all_images)_for_keypoints_detection_plus_descriptors_extraction,_by_detector_type_and_descriptor_type"
            << endl;
    write_descriptors_header();
    for (const auto &detector_name: detectorTypes) {
        image_detect_file << detector_name << " ";
        for (const auto &descriptor_name: descriptorTypes)
            image_detect_file << detect_extract_time[make_pair(detector_name, descriptor_name)] << " ";
        image_detect_file << endl;
    }
    return 0;
}