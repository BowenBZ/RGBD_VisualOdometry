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

#include <myslam/common_include.hpp>
#include <myslam/camera.hpp>

#include "myslam/private/mappoint.hpp"
#include "myslam/private/superpoint_model.hpp"

#include <optional>
#include <unordered_set>

namespace myslam 
{

class SuperPointModel;

struct FrameConfig {
    size_t      maxFeaturesCnt;    // max extracted features
    size_t      rowSectionCnt;     // how many section in rows for feature detection
    size_t      colSectionCnt;     // how many section in cols for feature detection

    size_t      imgCols;           // width of color image
    size_t      imgRows;           // height of color image

    size_t      gridSize;          // pixel's grid size
    size_t      gridColCnt;        // count of grid in image cols
    size_t      gridRowCnt;        // count of grid in image rows

    int         searchGridRadius;               // the radius of grid searching area
    double      descriptorDistanceThres;        // max distance between 2 descriptors to be considered as matched
    double      bestSecondaryDistanceRatio;     // min ratio between best match and secondary match to accept the best match 

    size_t      activeCovisibleWeight;          // threshold to set active covisible keyframe
};

typedef struct {
    // detected keypoints
    cv::KeyPoint keypoint;
    // extracted descriptors     
    cv::Mat descriptor;

    // Matched mappoint id. Only get populated if it's a keyframe
    // A feature point may not have matched mappoint if no matching found & no depth value found.
    std::optional<size_t> optMatchedMptId;
} KeypointInfo;

class Frame : public std::enable_shared_from_this<Frame>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Frame> Ptr;

    Camera::Ptr         camera_;        // Pinhole RGBD Camera model 

    // factory function
    static Frame::Ptr CreateFrame(
        const std::shared_ptr<struct FrameConfig> config,
        const double timestamp, 
        const Camera::Ptr& camera, 
        const cv::Mat& color, 
        const cv::Mat& depth
    ); 

    size_t GetId() const { 
        return id_; 
    }

    // Get T_c_w
    SE3 GetTcw() const {
        return T_c_w_;
    }

    void SetTcw(const SE3 pose) {
        T_c_w_ = std::move(pose);
    }

    // find the depth in depth map
    double GetDepth(const cv::KeyPoint& kp);
    
    // Release the RGB and depth image after temporary mappoint creation
    void ReleaseRawFrameData() {
        // OpenCV cv::Mat will automatically decrease the reference count
        color_ = cv::Mat();
        depth_ = cv::Mat();
    }

    // Get Camera Center
    Vector3d GetCamCenter() const {
        return T_c_w_.inverse().translation();
    }

#pragma mark - Feature extraction

    void ExtractKeyPointsAndComputeDescriptors(const cv::Ptr<cv::Feature2D>& detector);

    void ExtractKeypointsAndDescriptorsWithSuperPointModel(const std::shared_ptr<SuperPointModel> model);

    const size_t GetKeypointsSize() const {
        return keypointInfo_.size();
    }

    // Return the reference to keypoint
    const cv::KeyPoint& GetKeypoint(size_t idx) const {
        assert(idx < keypointInfo_.size());
        return keypointInfo_[idx].keypoint;
    }

    // Return the descriptor as a referene to the single row cv::Mat
    cv::Mat GetDescriptor(size_t idx) const {
        assert(idx < keypointInfo_.size());
        return keypointInfo_[idx].descriptor;
    }

    // Return all descriptors
    const cv::Mat& GetDescriptors() const {
        return descriptors_;
    }

#pragma mark - Feature matching

    // Get matched keypoint idx for the mappoint
    bool SearchKeypointMatchCandidate(const Mappoint::Ptr& mpt, const bool doDirectionCheck, size_t& kptIdx, double& distance, bool& mayObserveMpt);

#pragma mark - observing relationships

    /*
    * Add the observation relationship for existing mappoints already in MapManager
    * 1. Add observing mappoint (idx <-> mptId)
    * 2. Add observedBy keyframe to the mappoint, update the average descriptor of the mappoint
    * 3. Update the "only" observing mappoint of the anchor keyframe of this mappoint
    * 4. Update the covisible keyframes
    */ 
    void AddObservingMappointCreatedFromOtherFrame(const size_t kptIdx, const Mappoint::Ptr& mpt);

    /*
    * Add the observation relationship for new created mappoints from this frame
    * 1. Add observing mappoint (idx <-> mptId)
    * 2. Add "only" observing mappoint
    * 3. Add observedBy keyframe to the mappoint
    */
    void AddObservingMappointCreatedFromThisFrame(const size_t kptIdx, const Mappoint::Ptr& mpt);

    // Remove observed mappoint and also update the covisible keyframes
    void RemoveObservingMappointCreatedFromOtherFrame(const size_t mptId);

    // Remove observed mappoint created from this frame
    void RemoveObservingMappointCreatedFromThisFrame(const size_t mptId);

    const std::unordered_map<size_t, size_t>& GetAllObservingMptIdToKptIdx() {
        return observingMptIdToKptIdx_;
    }

    // Return if frame observes a mappoint
    bool IsObservingMappoint(const size_t id) {
        return observingMptIdToKptIdx_.count(id);
    }

    // Return the matched mappoint id matched with given keypoint index
    std::optional<size_t> GetMatchedMappointIdForKeypoint(const size_t kptIdx) {
        assert(kptIdx < keypointInfo_.size());
        return keypointInfo_[kptIdx].optMatchedMptId;
    }

    // Return the idx of the keypoint matched with given mappoint
    std::optional<size_t> GetMatchedKeypointIdxForMappoint(const size_t& mptId) {
        if (!observingMptIdToKptIdx_.count(mptId)) {
            return std::nullopt;
        }
        return observingMptIdToKptIdx_[mptId];
    }

#pragma mark - only observing relationships

    const std::unordered_set<size_t>& GetMappointIdsOnlyObservedByThisFrame() {
        return onlyThisObservedMptId_;
    }

    void AddOnlyThisObservedMpt(const size_t mptId) {
        onlyThisObservedMptId_.insert(mptId);
    }

    // Remove the "only" observation. Note this frame may still observe the mappoint
    void RemoveOnlyThisObservedMpt(const size_t mptId) {
        if (onlyThisObservedMptId_.count(mptId)) {
            onlyThisObservedMptId_.erase(mptId);
        }
    }

#pragma mark - covisible keyframes

    void GetActiveCovisibleKfIds(std::list<size_t>& activeCovisibleKfIds) {
        activeCovisibleKfIds.clear();
        activeCovisibleKfIds.insert(activeCovisibleKfIds.end(), activeCovisibleKfIds_.begin(), activeCovisibleKfIds_.end());
    }

    void GetAllCovisibleKfIds(std::list<size_t>& allCovisibleKfIds) {
        allCovisibleKfIds.clear();
        for (auto& [kfId, _]: allCovisibleKfIdToWeight_) {
            allCovisibleKfIds.push_back(kfId);
        }
    }

private: 
    static size_t           factoryId_;
    size_t                  id_;            // id of this frame
    double                  timestamp_;     // timestamp of RGB image

    std::shared_ptr<struct FrameConfig>  config_;

    cv::Mat                     color_;         // color image, become null after temporary mappoint creation
    cv::Mat                     depth_;         // depth image, become null after temporary mappoint creation

    SE3                     T_c_w_;         // transform from world to camera
    
    // detected feature points info
    std::vector<KeypointInfo>    keypointInfo_;
    // Each row is a descriptor
    cv::Mat                     descriptors_;

    std::unordered_map<size_t, std::list<size_t>> gridToKptIdx_;       // idx of keypoints for a grid

    // Only keyframe will use following fields. 
    // No need to add lock to the observation relationship, since frontend and backend won't modify the observation relationship the same time.

    // <mpt id, keypoint idx>
    std::unordered_map<size_t, size_t>   observingMptIdToKptIdx_;
    // id of mappoints only observed by this frame
    std::unordered_set<size_t>           onlyThisObservedMptId_;

    // <covisible keyframe id, weight>
    std::unordered_map<size_t, size_t>   allCovisibleKfIdToWeight_;
    // Active covisible keyframes ids (has same observed mappoints >= activeCovisibleWeight_)
    std::unordered_set<size_t>           activeCovisibleKfIds_;

    Frame(const std::shared_ptr<struct FrameConfig> config,
          const size_t id, 
          const double timestamp, 
          const Camera::Ptr& camera, 
          const cv::Mat& color, 
          const cv::Mat& depth);

    // Construct the keypoint grids for active search and match
    void ConstructKeypointGrids();

    // Calculate which grid the pixel point lies. (x: horizontal for col, y: vertical for row) is the pixel coordiante
    size_t GetGridIdx(double x, double y);

    // Get the grid idx given its col idx and row idx
    size_t GetGridIdx(size_t colIdx, size_t rowIdx);

    // Get nearby grid idx including the input grid
    void getNearbyGrids(size_t gridIdx, std::list<size_t>& nearbyGrids);

    // Update the covisible keyframe with new weight. Called by another object
    void UpdateCovisibleKeyframeWeight(const size_t otherKfId, const size_t weight);
};

}

#endif // FRAME_H
