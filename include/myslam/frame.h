/*
 * Reprensents a camera frame 
 */

#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/mappoint.h"

namespace myslam 
{

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

    SE3 SetPose(const SE3 pose) {
        unique_lock<mutex> lck(poseMutex_);
        T_c_w_ = move(pose);
    }
    
    // find the depth in depth map
    double GetDepth( const KeyPoint& kp );
    
    // Get Camera Center
    Vector3d GetCamCenter() const {
        return T_c_w_.inverse().translation();
    }
    
    // check if a point is in the view of this frame 
    bool IsCouldObserveMappoint( const Mappoint::Ptr& mpt );

    // Add observed mappoint and also update the covisible keyframes
    void AddObservedMappoint(const size_t mappointId, const Point2f pixelPos);

    // Remove observed mappoint and also update the covisible keyframes
    void RemoveObservedMappoint(const size_t mappointId);

    unordered_set<size_t> GetObservedMappointIds() {
        unique_lock<mutex> lck(observationMutex_);
        return observedMappointIds_;
    }

    bool IsObservedMappoint(const size_t id) {
        unique_lock<mutex> lck(observationMutex_);
        return observedMappointIds_.count(id);
    }

    // Update the covisible keyframe with new weight
    void UpdateCovisibleKeyframeWeight(const size_t id, const int weight);

    unordered_set<size_t> GetCovisibleKeyframes() {
        unique_lock<mutex> lck(observationMutex_);
        return activeCovisibleKeyframes_;
    }

private: 
    static size_t           factoryId_;
    size_t                  id_;         // id of this frame

    mutex                   poseMutex_;
    SE3                     T_c_w_;      // transform from world to camera

    mutex                   observationMutex_;
    unordered_set<size_t>   observedMappointIds_;
    CovisibleKeyframeIdToWeight allCovisibleKeyframeIdToWeight_;        // All covisible keyframes
    unordered_set<size_t>   activeCovisibleKeyframes_;     // Covisible keyframes (has same observed mappoints >= 15) and the number of covisible mappoints


    Frame(  const size_t id, 
            const double timestamp, 
            const Camera::Ptr camera, 
            const Mat color, 
            const Mat depth );

};

}

#endif // FRAME_H
