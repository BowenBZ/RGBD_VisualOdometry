/*
 * Reprensents a camera frame 
 */

#ifndef FRAME_H
#define FRAME_H

#include <opencv2/features2d/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/mappoint.h"

namespace myslam 
{

class Frame : public enable_shared_from_this<Frame>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<Frame> Ptr;
    typedef unordered_map<size_t, int> CovisibleKeyframeIdToWeight;

    double              timestamp_;     // when it is recorded
    Camera::Ptr         camera_;        // Pinhole RGBD Camera model 

    // factory function
    static Frame::Ptr CreateFrame(
        const double timestamp, 
        const Camera::Ptr& camera, 
        const Mat& color, 
        const Mat& depth
    ); 

    size_t GetId() const { 
        return id_; 
    }

    // Get T_c_w
    SE3 GetPose() {
        unique_lock<mutex> lck(poseMutex_);
        return T_c_w_;
    }

    void SetPose(const SE3 pose) {
        unique_lock<mutex> lck(poseMutex_);
        T_c_w_ = move(pose);
    }
    
    // Get the reference to the color image
    const Mat& GetImage() {
        return color_;
    }

    // find the depth in depth map
    double GetDepth( const KeyPoint& kp );
    
    // Get Camera Center
    Vector3d GetCamCenter() const {
        return T_c_w_.inverse().translation();
    }

    void ExtractKeyPointsAndComputeDescriptors(const cv::Ptr<cv::Feature2D>& detector);

    const size_t GetKeypointsSize() const {
        return keypoints_.size();
    }

    // Return the reference to keypoint
    const KeyPoint& GetKeypoint(size_t idx) const {
        return keypoints_[idx];
    }

    // Return the descriptor as a referene to the single row Mat
    Mat GetDescriptor(size_t idx) const {
        return descriptors_.row(idx);
    }

    // Return all descriptors
    const Mat& GetDescriptors() const {
        return descriptors_;
    }

    // Get matched keypoint idx for the mappoint
    bool GetMatchedKeypoint(const Mappoint::Ptr& mpt, size_t& kptIdx, double& distance, bool& mayObserveMpt);

    // 1. Add observing mappoint 
    // 2. Update the covisible keyframes
    // 3. Add observedBy keyframe to the mappoint
    void AddObservingMappoint(const Mappoint::Ptr& mpt, const size_t kptIdx);

    // Remove observed mappoint and also update the covisible keyframes
    void RemoveObservedMappoint(const size_t mappointId);

    unordered_set<size_t> GetObservedMappointIds() {
        unique_lock<mutex> lck(observationMutex_);
        return observingMappointIds_;
    }

    // Return if frame is already observeing mappoint
    bool IsObservingMappoint(const size_t id) {
        unique_lock<mutex> lck(observationMutex_);
        return observingMappointIds_.count(id);
    }

    // Update the covisible keyframe with new weight
    void UpdateCovisibleKeyframeWeight(const size_t id, const int weight);

    unordered_set<size_t> GetCovisibleKeyframes() {
        unique_lock<mutex> lck(observationMutex_);
        return activeCovisibleKeyframes_;
    }

private: 
    static size_t           factoryId_;
    size_t                  id_;            // id of this frame

    Mat                     color_;         // color image
    Mat                     depth_;         // depth image 

    mutex                   poseMutex_;
    SE3                     T_c_w_;         // transform from world to camera
    
    vector<KeyPoint>        keypoints_;     // detected keypoints
    Mat                     descriptors_;   // extracted descriptors

    size_t                  imgCols;                         // width of color image
    size_t                  imgRows;                         // height of color image
    size_t                  gridSize_;                       // pixel's grid size
    size_t                  gridColCnt_;                     // count of grid in image cols
    size_t                  gridRowCnt_;                     // count of grid in image rows
    unordered_map<size_t, list<size_t>> gridToKptIdx_;       // idx of keypoints for a grid

    int                     searchGridRadius_;               // the radius of grid searching area
    double                  descriptorDistanceThres_;        // max distance between 2 descriptors to be considered as matched
    double                  bestSecondaryDistanceRatio_;     // min ratio between best match and secondary match to accept the best match 

    mutex                   observationMutex_;
    unordered_set<size_t>   observingMappointIds_;
    CovisibleKeyframeIdToWeight allCovisibleKeyframeIdToWeight_;        // All covisible keyframes
    unordered_set<size_t>   activeCovisibleKeyframes_;     // Covisible keyframes (has same observed mappoints >= 15) and the number of covisible mappoints


    Frame(  const size_t id, 
            const double timestamp, 
            const Camera::Ptr& camera, 
            const Mat& color, 
            const Mat& depth );

    // Construct the keypoint grids for active search and match
    void ConstructKeypointGrids();

    // Calculate which grid the pixel point lies. (x: horizontal for col, y: vertical for row) is the pixel coordiante
    size_t GetGridIdx(double x, double y);

    // Get the grid idx given its col idx and row idx
    size_t GetGridIdx(size_t colIdx, size_t rowIdx);

    // Get nearby grid idx including the input grid
    vector<size_t> getNearbyGrids(size_t gridIdx);
};

}

#endif // FRAME_H
