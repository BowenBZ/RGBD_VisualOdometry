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
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/viewer.h"
#include "myslam/backend.h"

namespace myslam 
{
class Frontend
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<Frontend> Ptr;
    typedef unordered_map<size_t, Mappoint::Ptr> TrackingMap;

    enum VOState {
        INITIALIZING=0,
        TRACKING,
        LOST
    };

    Frontend(const Camera::Ptr& camera);
    
    bool AddFrame( const Frame::Ptr frame );      // entry point for application

    void SetViewer( const Viewer::Ptr viewer) {
        viewer_ = move(viewer);
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

    Camera::Ptr             camera_;
    Viewer::Ptr             viewer_;
    Backend::Ptr            backend_;

    VOState                 state_;             // current VO status
    size_t                  accuLostFrameNums_; // number of lost times

    Frame::Ptr              framePrev_;         // last frame
    Frame::Ptr              frameCurr_;         // current frame 
    
    TrackingMap             trackingMap_;  // the local tracking map

    cv::Ptr<cv::ORB>        orb_;               // orb detector and computer 
    cv::FlannBasedMatcher   flannMatcher_;      // flann matcher used if active search fails

    unordered_map<size_t, size_t>   matchedMptIdKptIdxMap_;     // matched mappoint id to keypoint idx
    unordered_map<size_t, size_t>   matchedKptIdxMptIdMap_;     // matched keypoint idx to mappoint id
    unordered_map<size_t, double>   matchedKptIdxDistanceMap_;  // matched keypoint idx to distance

    unordered_map<size_t, size_t>   flannMatchedMptIdKptIdxMap_;     // matched mappoint id to keypoint idx
    unordered_map<size_t, size_t>   flannMatchedKptIdxMptIdMap_;     // matched keypoint idx to mappoint id
    unordered_map<size_t, double>   flannMatchedKptIdxDistanceMap_;  // matched keypoint idx to distance

    unordered_map<size_t, size_t>   baInlierMptIdKptIdxMap_;      // inlier mappoint to kpt idx after PNP estimation
    unordered_set<size_t>   baInlierKptIdxSet_;     // inlier keypoint idx after PNP estimation
    size_t                  numInliers_;            // inlier count, should equal to size of baInlierKptIdxSet_
    
    g2o::SparseOptimizer    optimizer_;

    TrackingMap             lastFrameMpts_;         // mpt of last frame including matched mpts and temp mpts 
    unordered_map<Mappoint::Ptr, size_t> tempMptKptIdxMap_;   // temp mpts id to kpt idx

    // parameters, see config/default.yaml
    bool                    useActiveSearch_;       // if trying to use active search
    size_t                  minMatchesToUseFlannFrameTracking_;     // threshold to use flann for map matching
    size_t                  minMatchesToUseFlannMapTracking_;     // threshold to use flann for map matching
    float                   minDisRatio_;       // ratio for selecting flann good matches
    double                  baInlierThres_;     // threshold to be consider as an inlier after BA
    size_t                  minInliersForGood_; // minimum inliers to treat current frame as good
    size_t                  maxLostFrames_;     // max number of continuous lost times
    size_t                  minInliersForKeyframe_; // minimum inliers to consider current frame as keyframe
    double                  keyFrameMinRot_;    // minimal rotation of two key-frames
    double                  keyFrameMinTrans_;  // minimal translation of two key-frames
    
    mutex                   trackingMapMutex_;  // mutex for update tracking map

    void InitializationHandler();
    bool TrackingHandler();
    void LostHandler();

    // update tracking map, called by backend
    void UpdateTrackingMap(function<void(TrackingMap&)> updater);

    // match extracted features in tracking map
    void MatchKeyPointsWithMappoints(const TrackingMap& trackingMap, const bool doDirectionCheck, const size_t matchesToUseFlann);
    // match keypoints by flann
    void MatchKeyPointsFlann(const Mat& flannMptCandidateDes, unordered_map<int, size_t>& flannMptIdxToId);

    // estimate the pose with 3D-2D methods (mappoint, keypoint)
    void EstimatePoseMotionOnlyBA(TrackingMap& trackingMap); 

    // measure the estimation quality
    bool IsGoodEstimation(); 
    // determine whether treating as keyframe
    bool IsKeyframe();

    // create temp mappoints for current frame, used for next frame feature matching
    void CreateTempMappoints();
};
}

#endif // FrontEnd_H