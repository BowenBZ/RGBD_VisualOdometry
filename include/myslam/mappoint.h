/*
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
    bool        optimized_;             // whether is optimized by backend

    // Usages 
    // 1. whether match with new mappoints in front end
    // 2. whether will be triangulated
    // 3. whether add new observation keyframe 
    // 4. whether update in co-visibility graph
    // 5. whether add into backend
    bool        outlier_;               // whether this is an outlider
    
    // factory function to create mappoint
    // there will be only 1 time copy of parameters happening in the private constructor
    static Mappoint::Ptr CreateMappoint(const Vector3d& pos);

    Vector3d GetPosition() {
        unique_lock<mutex> lock(posMutex_);
        return pos_;
    }

    void SetPosition(const Vector3d pos) {
        unique_lock<mutex> lock(posMutex_);
        pos_ = move(pos);
    }

    size_t GetId() const { 
        return id_; 
    }

    // Set the descriptor for temp mappoints, it will be reset once observation is added
    void SetTempDescriptor(const Mat& descriptor) {
        descriptor_ = descriptor.clone();
    }

    Mat GetDescriptor() {
        unique_lock<mutex> lock(observationMutex_);
        return descriptor_;
    }

    Vector3d GetNormDirection() {
        unique_lock<mutex> lock(observationMutex_);
        return norm_;
    }

    void AddObservedByKeyframe(const shared_ptr<Frame>& kf, const size_t kptIdx);
    
    void RemoveObservedByKeyframe(const size_t kfId);

    unordered_map<size_t, size_t> GetObservedByKeyframesMap() {
        unique_lock<mutex> lock(observationMutex_);
        return observedByKfIdToKptIdx_;
    }


private:
    static size_t               factoryId_;
    size_t                      id_;

    Mat                         descriptor_;    // Descriptor for keypoint matching, coming from the best keypoint descriptor 
    Vector3d                    norm_;      // Normal of viewing direction 

    mutex                       posMutex_;
    Vector3d                    pos_;       // Position in world reference frame

    mutex                       observationMutex_;
    unordered_map<size_t, size_t>    observedByKfIdToKptIdx_;


    // mappoint can only be created by factory
    Mappoint(const size_t id, const Vector3d& pos);

    void CalculateMappointDescriptor();
};

} // namespace

#endif // MAPPOINT_H
