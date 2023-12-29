#include <gtest/gtest.h>
#include <iostream>
#include <list>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

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

// Demonstrate deep copy of Keypoint
TEST(BASISTEST, CopyOfKeyPoint) {
  cv::KeyPoint kpt1(1.0, 1,0, 0.1);
  cv::KeyPoint kpt2 = kpt1;

  EXPECT_EQ(kpt1.pt.x, kpt2.pt.x);
  kpt1.pt.x = 2.0;
  EXPECT_NE(kpt1.pt.x, kpt2.pt.x);
}

// Demonstrate deep copy of SE3
TEST(BASISTEST, CopyOfSE3) {
  Eigen::Matrix3d aR = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  Eigen::Vector3d aT (0, 0, 1);

  Sophus::SE3d a(aR, aT);
  Sophus::SE3d b = a;

  EXPECT_EQ(a.rotationMatrix(), b.rotationMatrix());
  a.setRotationMatrix(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)).toRotationMatrix());
  EXPECT_NE(a.rotationMatrix(), b.rotationMatrix());
}

// Demonstrate size of size_t
TEST(BASISTEST, SIZE_T) {
  EXPECT_EQ(sizeof(size_t), sizeof(size_t*));
}
