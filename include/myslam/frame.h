/*
 * Reprensents a frame
 * 
 * Provides following functions
 * 1. maintain the pose T_camera_from_world, detected features and descriptors
 * 2. active search to get a keypoint match for a mappoint
 * 3. for keyframe, maintain the observing mappoint and respective keypoint
 * 4. for keyframe, maintain the covisible graph with other keyframes
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
    SE3 GetTcw() {
        unique_lock<mutex> lck(poseMutex_);
        return T_c_w_;
    }

    void SetTcw(const SE3 pose) {
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
    bool GetMatchedKeypoint(const Mappoint::Ptr& mpt, const bool doDirectionCheck, size_t& kptIdx, double& distance, bool& mayObserveMpt);

    // 1. Add observing mappoint 
    // 2. Update the covisible keyframes
    // 3. Add observedBy keyframe to the mappoint
    void AddObservingMappoint(const Mappoint::Ptr& mpt, const size_t kptIdx);

    // Remove observed mappoint and also update the covisible keyframes
    void RemoveObservingMappoint(const size_t mptId);

    unordered_set<size_t> GetObservingMappointIds() {
        unique_lock<mutex> lck(observationMutex_);
        return observingMappointIds_;
    }

    // Return if frame is already observeing mappoint
    bool IsObservingMappoint(const size_t id) {
        unique_lock<mutex> lck(observationMutex_);
        return observingMappointIds_.count(id);
    }

    // Return if keypoint has matched mappoint
    bool IsKeypointMatchWithMappoint(const size_t kptIdx, size_t& mptId) {
        unique_lock<mutex> lck(observationMutex_);
        if (kptIdxToObservingMptIdMap_.count(kptIdx)) {
            mptId = kptIdxToObservingMptIdMap_[kptIdx];
            return true;
        }

        return false;
    }

    unordered_set<size_t> GetActiveCovisibleKfIds() {
        unique_lock<mutex> lck(observationMutex_);
        return activeCovisibleKfIds_;
    }

    unordered_set<size_t> GetAllCovisibleKfIds() {
        unique_lock<mutex> lck(observationMutex_);
        return allCovisibleKfIds_;
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
    unordered_map<size_t, size_t>  observingMptIdToKptIdxMap_;          // observing mpt to respective keypoint idx
    unordered_map<size_t, size_t>  kptIdxToObservingMptIdMap_;          // keypoint idx to respective mpt id

    unordered_map<size_t, size_t>   allCovisibleKfIdToWeight_;    // All covisible keyframe 
    unordered_set<size_t>           allCovisibleKfIds_;           // All covisible keyframe ids
    unordered_set<size_t>           activeCovisibleKfIds_;        // Active covisible keyframes (has same observed mappoints >= activeCovisibleWeight_) and the number of covisible mappoints
    size_t                          activeCovisibleWeight_;       // threshold to set active covisible keyframe


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

    // Update the covisible keyframe with new weight. Called by another object
    void UpdateCovisibleKeyframeWeight(const size_t otherKfId, const size_t weight);
};

}

#endif // FRAME_H
