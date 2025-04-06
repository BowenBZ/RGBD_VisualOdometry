/*
 * common include header files
 */


#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// define the commonly included file to avoid a long include std::list
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

#endif  // COMMON_INCLUDE_H
