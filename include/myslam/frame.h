/*
 */

#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/util.h"

namespace myslam 
{
    
// forward declare 
class Mappoint;

class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<Frame> Ptr;
    typedef unordered_map<size_t, int> CovisibleKeyframeIdToWeight;

    double              timestamp_;     // when it is recorded
    Camera::Ptr         camera_;        // Pinhole RGBD Camera model 
    Mat                 color_, depth_; // color and depth image 

    // factory function
    static Frame::Ptr CreateFrame(
        const double timestamp, 
        const Camera::Ptr camera, 
        const Mat color, 
        const Mat depth
    ); 

    size_t GetId() const { 
        return id_; 
    }

    SE3 GetPose() {
        unique_lock<mutex> lck(poseMutex_);
        return T_c_w_;
    }

    SE3 SetPose(const SE3& pose) {
        unique_lock<mutex> lck(poseMutex_);
        T_c_w_ = pose;
    }
    
    // find the depth in depth map
    double FindDepth( const KeyPoint& kp );
    
    // Get Camera Center
    Vector3d GetCamCenter() const;
    
    // check if a point is in this frame 
    bool IsInFrame( const Vector3d& pt_world );

    // Update the co-visible keyframes when this frame is a keyframe 
    void UpdateCovisibleKeyFrames();

    // Decrease the weight of connectedFrame by 1
    void DecreaseCovisibleKeyFrameWeightByOne(const size_t id);

    // Add the connection of another frame with weight to current frame
    void AddCovisibleKeyframe(const size_t id, const int weight) {
        unique_lock<mutex> lck(connectedMutex_);
        covisibleKeyframeIdToWeight_[id] = weight;
    }
    CovisibleKeyframeIdToWeight GetCovisibleKeyframes() {
        unique_lock<mutex> lck(connectedMutex_);
        return covisibleKeyframeIdToWeight_;
    }

    void AddObservedMappoint(const size_t id) {
        unique_lock<mutex> lck(observationMutex_);
        observedMappointIds_.insert(id);
    }

    void RemoveObservedMappoint(const size_t id);

    unordered_set<size_t> GetObservedMappointIds() {
        unique_lock<mutex> lck(observationMutex_);
        return observedMappointIds_;
    }

private: 
    static size_t           factoryId_;
    size_t                  id_;         // id of this frame

    mutex                   poseMutex_;
    SE3                     T_c_w_;      // transform from world to camera

    mutex                   connectedMutex_;
    CovisibleKeyframeIdToWeight covisibleKeyframeIdToWeight_;  // Covisible keyframes (has same observed mappoints >= 15) and the number of covisible mappoints

    mutex                   observationMutex_;
    unordered_set<size_t>   observedMappointIds_;

    Frame(  const size_t id, 
            const double timestamp, 
            const Camera::Ptr camera, 
            const Mat color, 
            const Mat depth );
};

}

#endif // FRAME_H
