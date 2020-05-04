#include "matching2D.hpp"
#include <opencv2/features2d.hpp>


using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource,
                      vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource,
                      cv::Mat &descRef,
                      vector<cv::DMatch> &matches,
                      string descriptorType,
                      string matcherType,
                      string selectorType) {
    if (descriptorType != "DES_BINARY" && descriptorType != "DES_HOG") {
        cerr << "unknown descriptor type: " << descriptorType;
        exit(1);
    }
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF") {
        // Configure a matcher for Brute-force matching
        bool crossCheck = false;
        int normType = descriptorType == "DES_HOG" ? cv::NORM_L2
                                                   : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType == "MAT_FLANN") {
        // Configure a matcher for FLANN matching
        // Workaround for bug in FLANN matcher, still present in OpenCV 4.3
        if (descSource.type() != CV_32F)
            descSource.convertTo(descSource, CV_32F);
        if (descRef.type() != CV_32F)
            descRef.convertTo(descRef, CV_32F);
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } else {
        cerr << "Unsupported matcher type: " << matcherType;
        exit(-1);
    }

    // Do the matching
    if (selectorType == "SEL_NN") {
        // Do it with nearest neighbor
        matcher->match(descSource, descRef, matches);
    } else if (selectorType == "SEL_KNN") {
        // Do it with k nearest neighbors (k=2)
        vector<vector<cv::DMatch> > knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        // Filter out matches that don't pass the Lowe's ratio test
        const float ratio_thresh = 0.8;
        for (auto & knn_match: knn_matches) {
            if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
                matches.push_back(knn_match[0]);
            }
        }
    } else {
        cerr << "Unsupported selector type: " << selectorType << endl;
        exit(-1);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType) {
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType == "BRISK") {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType == "ORB") {
        extractor = cv::ORB::create();
    } else if (descriptorType == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create();;
    } else if (descriptorType == "AKAZE") {
        extractor = cv::AKAZE::create();
    } else if (descriptorType == "SIFT") {
        extractor = cv::xfeatures2d::SIFT::create();;
    } else {
        cerr << "Unsupported descriptor: " << descriptorType << endl;
        exit(-1);
    }
    // perform feature description
    double t = (double) cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "  " << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}


void detGoodFeaturesToTrackHelper(vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &img,
                                  bool useHarrisDetector = false,
                                  bool bVis = false) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    auto t = static_cast<double>(cv::getTickCount());
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img,
                            corners,
                            maxCorners,
                            qualityLevel,
                            minDistance,
                            cv::Mat(),
                            blockSize,
                            false,
                            k);

    // add corners to result vector
    for (const auto &corner: corners) {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = static_cast<float>(blockSize);
        keypoints.push_back(newKeyPoint);
    }

    const string which_detection = useHarrisDetector ? "HARRIS" : "SHITOMASI";
    t = (static_cast<double >(cv::getTickCount()) - t) / cv::getTickFrequency();
    // cout << " " << which_detection << " with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = useHarrisDetector ? "Harris Corner Detector Results" : "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cout << "Press key to continue" << endl;
        cv::waitKey(0);
    }
}


void detGoodFeaturesToTrack(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    detGoodFeaturesToTrackHelper(keypoints, img, false, bVis);
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    detGoodFeaturesToTrackHelper(keypoints, img, true, bVis);
}

void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bViz) {

    auto t = (double) cv::getTickCount();

    cv::Ptr<cv::Feature2D> detector;

    if (detectorType == "FAST")
        detector = cv::FastFeatureDetector::create(10, true);
    else if (detectorType == "BRISK")
        detector = cv::BRISK::create();
    else if (detectorType == "ORB")
        detector = cv::ORB::create();
    else if (detectorType == "AKAZE")
        detector = cv::AKAZE::create();
    else if (detectorType == "SIFT")
        detector = cv::xfeatures2d::SIFT::create();

    detector->detect(img, keypoints);

    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << " " << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0
    //     << " ms"
    //     << endl;

// visualize results
    if (bViz) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}