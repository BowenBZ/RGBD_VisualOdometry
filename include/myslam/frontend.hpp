/*
 * Fontend which tracks camera poses on frames
 *
 * The entry point is the AddFrame function, which gets a frame pointer and computes the camera pose on that frame. Return false if the tracking fails.
 *
 * Frontend is reponsible for the other functions
 * 1. compute pose for new frame
 * 2. create temporary new mappoints for the frame
 * 3. determine if to create new keyframe 
 * 4. invoke backend to optimize new keyframe and new mappoints
 * 5. invoke reviewer (if there is) to show the image frames, real-time poses and maps
 */

#ifndef FrontEnd_H
#define FrontEnd_H

#include <myslam/common_include.hpp>
#include <myslam/camera.hpp>
#include <myslam/viewer.hpp>

#include <g2o/core/sparse_optimizer.h>

#include <functional>

namespace myslam 
{

class Frame;
struct FrameConfig;
class Mappoint;
class MapManager;
class SuperPointModel;
class Backend;
class UnaryEdgeProjection;

typedef struct {
    double timestamp;
    cv::Mat color;
    cv::Mat depth;
} Measurement;

typedef struct {
    float                   minDisRatio;       // Ratio for selecting flann good matches

    double                  baInlierThres;     // Threshold to be consider as an inlier after BA
    size_t                  minInliersForGood; // Minimum inliers to treat current frame as good
    size_t                  minInliersForKeyframe; // Minimum inliers to consider current frame as keyframe
    double                  maxFrameRotAllowed;    // minimal rotation of two key-frames
    double                  maxFrameTransAllowed;  // minimal translation of two key-frames

    size_t                  maxLostFrames;     // Max number of lost tracking frames
} FrontendConfig;

class Frontend
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Frontend> Ptr;
    typedef std::unordered_map<size_t, std::shared_ptr<Mappoint>> TrackingMap;
    typedef struct TrackingMapInfo {
        cv::Mat descriptors;
        std::vector<size_t> mptIds;

        void clear() {
            descriptors = cv::Mat();
            mptIds.clear();
        }
    } TrackingMapInfo;

    enum VOState {
        INITIALIZING=0,
        TRACKING,
        LOST
    };

    Frontend(const std::shared_ptr<Camera>& camera);
    
    // entry point for application
    bool AddFrame(const Measurement& measurement);

    // Get the latest pose
    SE3 GetPose();

    void SetViewer( const Viewer::Ptr viewer) {
        viewer_ = std::move(viewer);
    }

    VOState GetState() const { 
        return state_;
    }

    void Stop();
    
private:  
    const std::vector<std::string> VOStateStr {
        "Initializing", 
        "Tracking", 
        "Lost" 
    };                                          // used for logging

    std::shared_ptr<struct FrameConfig> frameConfig_;
    FrontendConfig          frontendConfig_;

    std::shared_ptr<Camera>             camera_;
    Viewer::Ptr                         viewer_;
    std::shared_ptr<Backend>            backend_;
    MapManager*                         mapManager_;
    std::shared_ptr<SuperPointModel>    superpointModel_;
    bool                                enableSuperpoint_;

    VOState                 state_;             // current VO status
    size_t                  accuLostFrameNums_; // number of lost times

    std::shared_ptr<Frame>  framePrev_;         // last frame
    std::shared_ptr<Frame>  frameCurr_;         // current frame 
    std::shared_ptr<Frame>  keyframeCurr_;      // current keyframe

    cv::Ptr<cv::ORB>        orb_;               // Orb detector and computer 
    cv::FlannBasedMatcher   flannMatcher_;      // flann matcher used if active search fails
    float                   nnThresh_;          // Threshold for NN matcher

    // mutex for update tracking map
    std::mutex              trackingMapMutex_;
    // the local tracking map sent from backend
    TrackingMap             localMap_;
    TrackingMapInfo         localMapInfo_;

    // New mappoints created from last frame 
    TrackingMap             lastFrameMap_;
    TrackingMapInfo         lastFrameMapInfo_;

    typedef struct {
        std::shared_ptr<Mappoint> mpt;
        float distance;
    } MatchInfo;
    // Matched (keypoint idx of current frame -> (mappoint id, distance))
    std::unordered_map<size_t, MatchInfo>   matchedKptIdxToInfo_;
    
    typedef struct {
        UnaryEdgeProjection *edge;
        bool isOutlier;
        size_t kptIdx;
    } EdgeInfo;

    g2o::SparseOptimizer    optimizer_;

    // (keypoint idx of current frame -> new created mappoints from current frame)
    std::unordered_map<size_t, std::shared_ptr<Mappoint>> kptIdxToNewMpt_;

    void InitializationHandler();
    bool TrackingHandler();
    void LostHandler();

    // update tracking map, called by backend
    void UpdateTrackingMap(std::function<void(TrackingMap&)> updater);

    /// Update the tracking map info when tracking is updated
    void UpdateTrackingMapInfo(const TrackingMap& trackingMap, TrackingMapInfo& info);

    // Find matched mappoints in tracking map for keypoints extracted from current frame
    void MatchKeyPointsWithMappoints(TrackingMap& trackingMap, TrackingMapInfo& info);

    /// Helper function to match current descriptors with tracking map
    /// @param trackingMap the mappoints to track against
    /// @param trackingMapInfo metadata of mappoints
    /// @param currDescriptors descriptors of current frame
    /// @param nnThresh max distance threshold for a matched keypoint and mappoint
    /// @param matchedKptIdxToInfo matched keypoint to its match info
    void FindMatch(TrackingMap& trackingMap, TrackingMapInfo& trackingMapInfo, cv::Mat currDescriptors, const float nnThresh, std::unordered_map<size_t, MatchInfo>& matchedKptIdxToInfo);

    /// Match current frame's keypoints with tracking map and then last frame
    void MatchKeyPointsWithTrackingMapAndLastFrameNN();

    // Estimate the pose with 3D-2D methods (mappoint, keypoint)
    void EstimateCurrentFramePose(const bool doMotionBA); 

    // measure the estimation quality
    bool IsGoodEstimation(); 
    // determine whether treating as keyframe
    bool IsKeyframe();

    // create temp mappoints for current frame, used for next frame feature matching
    void CreateTempMappoints();

    // Send keyframe to backend
    void SendKeyframeToBackend();
};
}

#endif // FrontEnd_H