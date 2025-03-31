/*
 * common include header files
 */


#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix4f;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// for Sophus
#include <sophus/se3.hpp>
typedef Sophus::SE3d SE3;

// #include "myslam/math/se3.hpp"
// typedef myslam::math::SE3<double> SE3;

// for cv
#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::KeyPoint;
using cv::Point3f;
using cv::Point2f;

// std 
#include <iostream>
#include <cstdio>
#include <list>
#include <queue>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <typeinfo>
#include <assert.h>

// #ifdef NDEBUG
// #undef assert
// #define assert(condition) ((void)0)
// #endif

using namespace std; 

#endif  // COMMON_INCLUDE_H
