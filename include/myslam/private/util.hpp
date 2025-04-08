#ifndef MYSLAM_UTIL_H
#define MYSLAM_UTIL_H

#include <myslam/common_include.hpp>

#include <unordered_set>

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param normalizedMptPos    normalizedMptPos in normalized plane
 * @param mptPosWorld  triangulated point in the world
 * @return true if success
 */
inline bool Triangulation(const std::vector<SE3>&        poses,
                          const std::vector<Vector3d>&   normalizedMptPos, 
                          Vector3d&                 mptPosWorld) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(2 * poses.size(), 4);
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = normalizedMptPos[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = normalizedMptPos[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    mptPosWorld = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true;
    }
    return false;
}

/*
Find the matched keypoints by finding the minimum L2 distance, i.e.
distance = \sqrt{ \sum_i^256 (x_i - y_i)^2 }
         = \sqrt{ \sum_i^256 (x_i^2 + y_i^2 - 2 x_i y_i) }
         = \sqrt{ 2 - 2 * x \dot y}

Input:
- prev_desc (N1, 256)
- curr_desc (N2, 256)

Output:
- matched index of curr_desc
- respective index of prev_desc
*/
inline void find_matched_points(const cv::Mat& prev_desc,
                                const cv::Mat& curr_desc,
                                const float nn_thresh,
                                std::vector<int>& matchedCurrentIndices,
                                std::vector<int>& matchedPrevIndices,
                                std::vector<float>& matchedDistance) {
    matchedCurrentIndices.clear();
    matchedPrevIndices.clear();

    // If no previous descriptors, return empty matched points and all current points as unmatched.
    if (prev_desc.empty()) {
        return;
    }

    // Dimensions:
    int N1 = prev_desc.rows;
    int N2 = curr_desc.rows;

    cv::Mat curr_desc_T;
    cv::transpose(curr_desc, curr_desc_T);
    // (N1, N2)
    cv::Mat dot_product = prev_desc * curr_desc_T;

    // Compute L2 distance = sqrt(2 - 2 * dot_product) element-wise.
    // Use element-wise operations.
    cv::Mat distance = 2 - 2 * dot_product;
    cv::sqrt(distance, distance);

    // For each previous descriptor (row of 'distance'), find the current descriptor index with the minimum distance.
    std::vector<int> matched_curr_indices(N1, -1);
    for (int i = 0; i < N1; i++) {
        float min_val = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j = 0; j < N2; j++) {
            float val = distance.at<float>(i, j);
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        matched_curr_indices[i] = min_idx;
    }

    // For each current descriptor (column of 'distance'), find the previous descriptor index with the minimum distance.
    std::vector<int> matched_prev_indices(N2, -1);
    for (int j = 0; j < N2; j++) {
        float min_val = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int i = 0; i < N1; i++) {
            float val = distance.at<float>(i, j);
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        matched_prev_indices[j] = min_idx;
    }

    // Create a std::vector of current point indices: 0, 1, ..., N2-1.
    // Compute bidirectional matches: for each current descriptor j, check if:
    //   j == matched_curr_indices[ matched_prev_indices[j] ]
    std::vector<bool> bidirectional_match(N2, false);
    for (int j = 0; j < N2; j++) {
        int prev_idx = matched_prev_indices[j];
        if (prev_idx >= 0 && prev_idx < N1) {
            bidirectional_match[j] = (j == matched_curr_indices[prev_idx]);
        }
    }

    // For each current descriptor j, check if its matching score is below the threshold.
    std::vector<bool> pass_thresh_match(N2, false);
    for (int j = 0; j < N2; j++) {
        int prev_idx = matched_prev_indices[j];
        float score = distance.at<float>(prev_idx, j);
        pass_thresh_match[j] = (score < nn_thresh);
    }

    // Final match: for each current point j, it is a valid match if both conditions are true.
    for (int j = 0; j < N2; j++) {
        if (bidirectional_match[j] && pass_thresh_match[j]) {
            matchedCurrentIndices.push_back(j);
            int prev_idx = matched_prev_indices[j];
            matchedPrevIndices.push_back(prev_idx);
            float score = distance.at<float>(prev_idx, j);
            matchedDistance.push_back(score);
        }
    }
}

inline Vector2d toVector2d(const cv::Point2f& pt) {
    return Vector2d ( pt.x, pt.y );
}

inline Vector2d toVector2d(const cv::KeyPoint& kp) {
    return toVector2d( kp.pt );
}

inline Vector3d toVector3d(const cv::Point3f& pt) {
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

typedef std::unordered_set<cv::KeyPoint, KeyPointHash, KeyPointsComparision> KeyPointSet;

// Compute the Hamming distance between 2 descriptors
// Descriptor is provided as a row in the cv::Mat
inline double ComputeDescriptorHammingDistance(
    const cv::Mat& desMat1, size_t row1,
    const cv::Mat& desMat2, size_t row2) {

    assert(desMat1.cols == desMat2.cols);

    auto& cols = desMat1.cols;
    double distance = 0;
    for(size_t col = 0; col < cols; ++col) {
        distance += (desMat1.at<unsigned char>(row1, col) !=
                     desMat2.at<unsigned char>(row2, col));
    }

    return distance;
}

// Compute the Hamming distance between 2 descriptors
// Descriptor is provided as a row in the cv::Mat
inline double ComputeSuperpointDescriptorL2Distance(const cv::Mat& desMat1, const cv::Mat& desMat2) {

    assert(desMat1.cols == desMat2.cols);
    assert(desMat1.rows == 1);
    assert(desMat2.rows == 1);

    // Compute L2 distance = sqrt(2 - 2 * dot_product) element-wise.

    cv::Mat desMat2_T;
    cv::transpose(desMat2, desMat2_T);
    // (1xN) x (Nx1)
    cv::Mat dot_product = desMat1 * desMat2_T;
    
    // Use element-wise operations.
    cv::Mat distance = 2 - 2 * dot_product;
    cv::sqrt(distance, distance);

    return distance.at<float>(0, 0);
}

} // namespace

#endif  // MYSLAM_UTIL_H