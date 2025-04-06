/*
 * Reprensents a mappoint
 * 
 * Provides following functions
 * 1. maintain the 3d position in world frame, and descriptor
 * 2. maintain the observedBy keyframe and the respective keypoint idx in that keyframe
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{

class Frame;

class Mappoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Mappoint> Ptr;

    bool        triangulated_;          // whether have been triangulated in frontend

    // If this mappoint is optimized as inlier by backend
    // only optimized and inlier mpt can be used as tracking map for frontend
    bool        optimized_;

    // If this mappoint is not observed by any keyframe
    // Outlier mappoint cannot 
    // 1. used as a point in tracking map in frontend
    // 2. matched with new keyframe
    // 3. be triangulated or optimized
    // 4. should be removed from map manager
    bool        outlier_;
    
    // factory function to create mappoint
    // there will be only 1 time copy of parameters happening in the private constructor
    static Mappoint::Ptr CreateMappoint(const Vector3d& pos, const cv::Mat& descriptor);

    const Vector3d& GetPosition() const {
        return pos_;
    }

    void SetPosition(const Vector3d pos) {
        pos_ = std::move(pos);
    }

    const size_t& GetId() const { 
        return id_; 
    }

    // Recalculate the norm. Only needed to be called when mpt position changes, or observedBy keyframe is removed or pose changes
    void UpdateNormViewDirection();

    // Recalculate descriptor when it's observed by several keyframes
    void UpdateDescriptor();

    const cv::Mat& GetDescriptor() {
        return descriptor_;
    }

    Vector3d GetNormDirection() {
        return norm_;
    }

#pragma mark - Observation relationships

    // only be called by keyframe object
    void AddObservedByKeyframe(const std::shared_ptr<Frame>& kf);
    
    // only be called by keyframe object
    void RemoveObservedByKeyframe(const size_t kfId);

    unordered_set<size_t>& GetObservedByKeyframeIds() {
        return observedByKfId_;
    }

    void AddAnchoringKeyframeId(const size_t keyframeId) {
        anchorKfId_ = keyframeId;
    }

    const size_t& GetAnchoringKeyframeId() const {
        return anchorKfId_;
    }

private:
    static size_t               factoryId_;
    size_t                      id_;

    // Descriptor for keypoint matching, coming from the best keypoint descriptor 
    cv::Mat                         descriptor_;
    // Normal of viewing direction, from mappoint to camera
    Vector3d                    norm_;

    // Position in world reference frame
    Vector3d                    pos_;           

    // No need to add lock since frontend and backend won't update at the same time.
    unordered_set<size_t>       observedByKfId_;

    // The keyframe this mappoint is created from
    size_t                      anchorKfId_;

    // mappoint can only be created by factory
    Mappoint(const size_t id, const Vector3d& pos, const cv::Mat& descriptor);

};

} // namespace

#endif // MAPPOINT_H
