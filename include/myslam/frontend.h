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

#include <opencv2/features2d/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/private/frame.h"
#include "myslam/private/mappoint.h"
#include "myslam/private/superpoint_model.hpp"
#include "myslam/viewer.h"
#include "myslam/private/backend.h"

namespace myslam 
{

typedef struct {
    double timestamp;
    Mat color;
    Mat depth;
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
    typedef unordered_map<size_t, Mappoint::Ptr> TrackingMap;

    enum VOState {
        INITIALIZING=0,
        TRACKING,
        LOST
    };

    Frontend(const Camera::Ptr& camera);
    
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
    const vector<string> VOStateStr {
        "Initializing", 
        "Tracking", 
        "Lost" 
    };                                          // used for logging

    FrameConfig             frameConfig_;
    FrontendConfig          frontendConfig_;

    Camera::Ptr             camera_;
    Viewer::Ptr             viewer_;
    Backend::Ptr            backend_;
    MapManager::Ptr         mapManager_;
    SuperPointModel::Ptr    superpointModel_;
    bool                    enableSuperpoint_;

    VOState                 state_;             // current VO status
    size_t                  accuLostFrameNums_; // number of lost times

    Frame::Ptr              framePrev_;         // last frame
    Frame::Ptr              frameCurr_;         // current frame 
    Frame::Ptr              keyframeCurr_;      // current keyframe

    cv::Ptr<cv::ORB>        orb_;               // Orb detector and computer 
    cv::FlannBasedMatcher   flannMatcher_;      // flann matcher used if active search fails
    float                   nnThresh_;          // Threshold for NN matcher

    // mutex for update tracking map
    mutex                   trackingMapMutex_;
    // the local tracking map sent from backend
    TrackingMap             trackingMap_;       
    // Mappoints observed by last frame, including matched mappoints from trackingMap_ and new mappoints created from last frame 
    TrackingMap             lastFrameMpts_;

    typedef struct {
        Mappoint::Ptr mpt;
        float distance;
    } MatchInfo;
    // Matched (keypoint idx of current frame -> (mappoint id, distance))
    unordered_map<size_t, MatchInfo>   matchedKptIdxToInfo_;
    
    g2o::SparseOptimizer    optimizer_;

    // (keypoint idx of current frame -> new created mappoints from current frame)
    unordered_map<size_t, Mappoint::Ptr> kptIdxToNewMpt_;

    void InitializationHandler();
    bool TrackingHandler();
    void LostHandler();

    // update tracking map, called by backend
    void UpdateTrackingMap(function<void(TrackingMap&)> updater);

    // Find matched mappoints in tracking map for keypoints extracted from current frame
    void MatchKeyPointsWithMappoints(TrackingMap& trackingMap);

    // Find matched mappints in tracking map for keypoints extracted from current frame using NN match
    void MatchKeyPointsWithMappointsNN(TrackingMap& trackingMap);

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