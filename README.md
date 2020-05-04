# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies
* cmake >= 3.1
* make >= 4.1 
* OpenCV >= 4.1
* gcc/g++ >= 5.4

Tested under Ubuntu 16.04 (OpenCV 4.1) and Ubuntu 18.04 (OpenCV 4.3)

## Build Instructions

1. Clone this repo: `git clone https://github.com/fantauzzi/SFND_2D_Feature_Tracking.git`
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Build: `cmake .. && make`
4. Run it: `./2D_feature_tracking`

## Performance Evaluation

Performance using different keypoint detectors and their descriptors have been measured, and are reported below. Where applicable, the keypoint matching algorithm used is FLANN, with k-nearest neighbor selection (k=2).

The table below indicates the number of keypoints detected on the preceding vehicle, for each image and for each detector. Shi-Tomasi and Harris detect the fewest, in every image, while Fast detect the most.

The table below lists the average neighborhood size of the keypoints on every image, for the different detectors. The next table lists the sample standard deviation. Shi-Tomasi, Harris and Fast have a fixed neighborhood size for every keypoint, therefore have, across images, the same average and zero sample standard deviation.

The total number of keypoints matched between two consecutive images is in the table below, summed across 9 pairs of consecutive images.

This is the total time for keypoints detection and description extraction in milliseconds, summed across the 10 images.

The last table reports the number of matched keypoints between two consecutive images, divided by the time needed for keypoints detection and description extraction.

## Rubric Criteria

* **MP.0 Mid-Term Report** -This file.
* **MP.1 Data Buffer Optimization** -Implemented as a vector of fixed size, with an index (`pos_in_buffer`) that points at the image currently being processed; the index is incremented to point to the next image, and when the index points past the last element in the vector, it is reset to 0.
* **MP.2 Keypoint Detection** -Implemented in `matching2D_Student.cpp`.
* **MP.3 Keypoint Removal** -Implemented in `MidTermProject_Camera_Student`.
* **MP.4 Keypoint Descriptors** -Implemented in `matching2D_Student.cpp`.
* **MP.5 Descriptor Matching** -Implemented in `matching2D_Student.cpp`.
* **MP.6 Descriptor Distance Ratio** -Implemented in `matching2D_Student.cpp`.
* **MP.7 Performance Evaluation 1** -See previous section, the first three tables.
* **MP.8 Performance Evaluation 2** -See previous section, the fourth table.
* **MP.9 Performance Evaluation 3** -See previous section, the last table and conclusions thereafter.
 