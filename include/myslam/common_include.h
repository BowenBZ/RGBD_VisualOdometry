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

// opencv
#include <opencv2/opencv.hpp>

// for Sophus
#include <sophus/se3.hpp>
typedef Sophus::SE3d SE3;

// #include "myslam/math/se3.hpp"
// typedef myslam::math::SE3<double> SE3;

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

using namespace std;

#endif  // COMMON_INCLUDE_H
