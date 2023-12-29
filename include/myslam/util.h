#ifndef MYSLAM_UTIL_H
#define MYSLAM_UTIL_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool Triangulation(const vector<SE3>&    poses,
                          const vector<Vec3>&   points, 
                          Vec3&                 pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true;
    }
    return false;
}

inline Vector2d toVec2d(const Point2f& pt) {
    return Vector2d ( pt.x, pt.y );
}

inline Vector2d toVec2d(const KeyPoint& kp) {
    return toVec2d( kp.pt );
}

inline Vector3d toVec3d(const Point3f& pt) {
    return Vector3d ( pt.x, pt.y, pt.z );
}

inline Point3f toPoint3f(const Vector3d& pt) {
    return Point3f( pt(0,0), pt(1,0), pt(2,0) );
}

struct KeyPointHash   
{  
    size_t operator()(const KeyPoint& kpt) const  
    {  
        return kpt.hash();  
    }  
};

struct KeyPointsComparision  
{  
    bool operator()(const KeyPoint& kpt1, const KeyPoint& kpt2) const  
    {  
        return kpt1.hash() == kpt2.hash();  
    }  
};

typedef unordered_set<KeyPoint, KeyPointHash, KeyPointsComparision> KeyPointSet;

// Compute the Hamming distance between 2 descriptors
// Descriptor is provided as a row in the Mat
inline double ComputeDescriptorDistance(
    const Mat& desMat1, size_t row1,
    const Mat& desMat2, size_t row2) {

    assert(desMat1.cols == desMat2.cols);

    auto& cols = desMat1.cols;
    double distance = 0;
    for(size_t col = 0; col < cols; ++col) {
        distance += (desMat1.at<unsigned char>(row1, col) !=
                     desMat2.at<unsigned char>(row2, col));
    }

    return distance;
}


} // namespace

#endif  // MYSLAM_UTIL_H