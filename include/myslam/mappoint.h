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

    typedef shared_ptr<Mappoint> Ptr;

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
    static Mappoint::Ptr CreateMappoint(const Vector3d& pos, const Mat& descriptor);

    Vector3d GetPosition() {
        unique_lock<mutex> lock(posMutex_);
        return pos_;
    }

    void SetPosition(const Vector3d pos) {
        unique_lock<mutex> lock(posMutex_);
        pos_ = move(pos);
    }

    const size_t& GetId() const { 
        return id_; 
    }

    // Recalculate the norm. Only needed to be called when mpt position changes, or observedBy keyframe is removed or pose changes
    void UpdateNormViewDirection();

    // Recalculate descriptor when it's observed by several keyframes
    void UpdateDescriptor();

    Mat GetDescriptor() {
        unique_lock<mutex> lock(observationMutex_);
        return descriptor_;
    }

    Vector3d GetNormDirection() {
        unique_lock<mutex> lock(observationMutex_);
        return norm_;
    }

    // only be called by keyframe object
    void AddObservedByKeyframe(const shared_ptr<Frame>& kf, const size_t kptIdx);
    
    // only be called by keyframe object
    void RemoveObservedByKeyframe(const size_t kfId);

    unordered_map<size_t, size_t> GetObservedByKeyframesMap() {
        unique_lock<mutex> lock(observationMutex_);
        return observedByKfIdToKptIdx_;
    }


private:
    static size_t               factoryId_;
    size_t                      id_;

    Mat                         descriptor_;    // Descriptor for keypoint matching, coming from the best keypoint descriptor 
    Vector3d                    norm_;          // Normal of viewing direction, from mappoint to camera

    mutex                       posMutex_;
    Vector3d                    pos_;           // Position in world reference frame

    mutex                       observationMutex_;
    unordered_map<size_t, size_t>    observedByKfIdToKptIdx_;

    // mappoint can only be created by factory
    Mappoint(const size_t id, const Vector3d& pos, const Mat& descriptor);

};

} // namespace

#endif // MAPPOINT_H
