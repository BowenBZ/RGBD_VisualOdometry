#include <gtest/gtest.h>
#include <iostream>
#include <list>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

// Demonstrate deep copy of Vector3d
TEST(BASISTEST, CopyOfVector3d) {
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d b = a;
  EXPECT_EQ(a, b);

  a[0] = 4;
  
  EXPECT_NE(a, b);
}

// Demonstrate deep copy of Point2f
TEST(BASISTEST, CopyOfPoint2f) {
  cv::Point2f a (1, 2);
  cv::Point2f b = a;
  EXPECT_EQ(a, b);

  a.x = 4;
  
  EXPECT_NE(a, b);
}

// Demonstrate deep copy of Point2f
TEST(BASISTEST, MakeCopyOfPoint2f) {
  cv::Point2f a (1, 2);
  std::list<std::pair<size_t, cv::Point2f>> b;
  b.push_back(std::make_pair(0, a));

  EXPECT_EQ(a, b.front().second);

  a.x = 4;
  
  EXPECT_NE(a, b.front().second);
}

// Demonstrate shadow copy of Mat
TEST(BASISTEST, CopyOfMat_changeOrigin) {
  cv::Mat a = (cv::Mat_<double>(2, 2) << 1, 2, 3, 4);
  cv::Mat b = a;
  EXPECT_EQ(a.at<double>(0, 0), b.at<double>(0, 0));
  
  a.at<double>(0, 0) = 5;

  EXPECT_EQ(a.at<double>(0, 0), b.at<double>(0, 0));
}

// Demonstrate shadow copy of Mat 2
TEST(BASISTEST, CopyOfMat_originReset) {
  cv::Mat a = (cv::Mat_<double>(2, 2) << 1, 2, 3, 4);
  cv::Mat b = a;
  EXPECT_EQ(a.at<double>(0, 0), b.at<double>(0, 0));

  a = (cv::Mat_<double>(2, 2) << 4, 3, 2, 1);
   
  EXPECT_NE(a.at<double>(0, 0), b.at<double>(0, 0));
}

class MatTestClass {
  public:
    cv::Mat mem;
    MatTestClass(cv::Mat input): mem(input) {

    }
};

// Demonstrate shadow copy of Mat 3
TEST(BASISTEST, PassOfMat_originReset) {
  cv::Mat a = (cv::Mat_<double>(2, 2) << 1, 2, 3, 4);
  MatTestClass b(a);
  EXPECT_EQ(a.at<double>(0, 0), b.mem.at<double>(0, 0));

  a.at<double>(0, 0) = 5;
   
  EXPECT_EQ(a.at<double>(0, 0), b.mem.at<double>(0, 0));
}
