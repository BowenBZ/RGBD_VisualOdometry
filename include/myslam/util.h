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
inline bool triangulation(const vector<SE3> &poses,
                          const vector<Vec3> points, 
                          Vec3 &pt_world) {
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

inline Vector2d toVec2d(const cv::Point2f& pt) {
    return Vector2d ( pt.x, pt.y );
}

inline Vector2d toVec2d(const cv::KeyPoint& kp) {
    return toVec2d( kp.pt );
}

inline Vector3d toVec3d(const cv::Point3f& pt) {
    return Vector3d ( pt.x, pt.y, pt.z );
}

inline cv::Point3f toPoint3f(const Vector3d& pt) {
    return cv::Point3f( pt(0,0), pt(1,0), pt(2,0) );
}

struct KeyPointHash   
{  
    size_t operator()(const cv::KeyPoint& kpt) const  
    {  
        return kpt.hash();  
    }  
};

struct KeyPointsComparision  
{  
    bool operator()(const cv::KeyPoint& kpt1, const cv::KeyPoint& kpt2) const  
    {  
        return kpt1.hash() == kpt2.hash();  
    }  
};

typedef unordered_set<cv::KeyPoint, KeyPointHash, KeyPointsComparision> KeyPointSet;


template<class T>
struct WeakPtrHash
{  
    size_t operator()(const weak_ptr<T>& ptr) const  
    {  
        return (ptr.expired()) ? 0: hash<shared_ptr<T>>()(ptr.lock());
    }  
};

template<class T>
struct WeakPtrComparision
{  
    bool operator()(const weak_ptr<T>& ptr1, const weak_ptr<T>& ptr2) const  
    {  
        if (!ptr1.expired() && !ptr2.expired()) {
            return hash<shared_ptr<T>>()(ptr1.lock()) == hash<shared_ptr<T>>()(ptr2.lock());
        } else if (ptr1.expired() && ptr2.expired()) {
            return true;
        } else {
            return false;
        }
    }  
};


} // namespace

#endif  // MYSLAM_UTIL_H