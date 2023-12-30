/*
 * Frontend for tracking
 */

#ifndef FrontEnd_H
#define FrontEnd_H

#include <opencv2/features2d/features2d.hpp>
#include <bits/stdc++.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/viewer.h"
#include "myslam/backend.h"
#include "myslam/util.h"

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

    Frontend();
    
    bool AddFrame( const Frame::Ptr frame );      // entry point for application

    void SetViewer( const Viewer::Ptr viewer) {
        viewer_ = move(viewer);
    }

    void SetBackend( const Backend::Ptr backend) {
        backend_ = move(backend);

        backend_->RegisterTrackingMapUpdateCallback(
            [&](function<void(Frame::Ptr&, TrackingMap&)> updater) {
                UpdateTrackingMap(updater);
            });
    }

    VOState GetState() const { 
        return state_;
    }
    
private:  
    const vector<string> VOStateStr {
        "Initializing", 
        "Tracking", 
        "Lost" 
    };                                          // used for logging

    Viewer::Ptr             viewer_;
    Backend::Ptr            backend_;

    VOState                 state_;             // current VO status
    size_t                  accuLostFrameNums_; // number of lost times

    Frame::Ptr              framePrev_;         // last frame
    Frame::Ptr              frameCurr_;         // current frame 
    
    TrackingMap             trackingMap_;  // the local tracking map
    Frame::Ptr              keyframeForTrackingMap_;    // the keyframe which is used to identify the tracking map

    cv::Ptr<cv::ORB>        orb_;               // orb detector and computer 
    cv::FlannBasedMatcher   flannMatcher_;      // flann matcher used if active search fails

    unordered_map<size_t, size_t>   matchedMptIdKptIdxMap_;     // matched mappoint id to keypoint idx
    unordered_map<size_t, size_t>   matchedKptIdxMptIdMap_;     // matched keypoint idx to mappoint id
    unordered_map<size_t, double>   matchedKptIdxDistanceMap_;  // matched keypoint idx to distance

    unordered_set<size_t>   baInlierMptIdSet_;      // inlier mappoint id after PNP estimation
    unordered_set<size_t>   baInlierKptIdxSet_;     // inlier keypoint idx after PNP estimation
    size_t                  numInliers_;            // inlier count, should equal to size of baInlierMptIdSet_ and baInlierKptIdxSet_
    
    g2o::SparseOptimizer    optimizer_;

    TrackingMap             lastFrameMpts_;         // mpt of last frame including matched mpts and temp mpts 
    list<Mappoint::Ptr>     tempMpts_;                      // temp mpts created from current frame
    unordered_map<size_t, size_t> tempMptIdToKptIdx_;       // temp mpts id to kpt idx

    // parameters, see config/default.yaml
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
    void UpdateTrackingMap(function<void(Frame::Ptr&, TrackingMap&)> updater);

    // match extracted features in tracking map
    void MatchKeyPointsWithMappoints(const TrackingMap& trackingMap, size_t matchesToUseFlann);
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
    // create mappoints from new observed keypoint of current frame
    void AddTempMappointsToMapManager();
    // add observing mappoints (both previous and new created) to current keyframe
    void AddObservingMappointsToCurrentFrame();
    // add new mappoints to the observedMappoints of existing keyframes in tracking map
    void AddNewMappointsObservationsForOldKeyframes();
};
}

#endif // FrontEnd_H