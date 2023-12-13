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
class FrontEnd
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<FrontEnd> Ptr;
    enum VOState {
        INITIALIZING=0,
        TRACKING,
        LOST
    };

    FrontEnd();
    
    bool AddFrame( const Frame::Ptr frame );      // entry point for application

    void SetViewer( const Viewer::Ptr viewer) {
        viewer_ = move(viewer);
    }

    void SetBackend( const Backend::Ptr backend) {
        backend_ = move(backend);
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
    int                     accuLostFrameNums_; // number of lost times

    Frame::Ptr              keyframeRef_;       // reference keyframe, used for getting local tracking map
    Frame::Ptr              framePrev_;         // last frame
    Frame::Ptr              frameCurr_;         // current frame 
    
    unordered_map<size_t, Mappoint::Ptr> trackingMap_;  // the local tracking map
    Frame::Ptr              keyframeForTrackingMap_;    // the keyframe which is used to identify the tracking map

    cv::Ptr<cv::ORB>        orb_;               // orb detector and computer 
    vector<KeyPoint>        keypointsCurr_;     // keypoints in current frame
    Mat                     descriptorsCurr_;   // descriptor in current frame 
    cv::FlannBasedMatcher   flannMatcher_;      // flann matcher
    unordered_map<Mappoint::Ptr, KeyPoint>  flannMatchedMptKptMap_;   // matched map points and keypoints after flann
    KeyPointSet             matchedKptSet_;     // set of matched keypoint

    unordered_set<Mappoint::Ptr> pnpMatchedMptSet_;    // matched mappoints set after PNP estimation
    
    vector<Mappoint::Ptr>   newMappoints_;      // new mappoints created for new keyframe

    int                     numInliers_;        // number of inlier features in pnp estimation
    
    // parameters, see config/default.yaml
    float                   minDisRatio_;       // ratio for selecting good matches
    int                     maxLostFrames_;     // max number of continuous lost times
    int                     minInliers_;        // minimum inliers
    double                  keyFrameMinRot_;    // minimal rotation of two key-frames
    double                  keyFrameMinTrans_;  // minimal translation of two key-frames
    
    void InitializationHandler();
    bool TrackingHandler();
    void LostHandler();

    // extract features from current frame
    void ExtractKeyPointsAndComputeDescriptors();
    // match extracted features in tracking map
    void MatchKeyPointsInTrackingMap();
    // estimate the pose with 3D-2D methods (mappoint, keypoint)
    void EstimatePosePnP(); 

    // measure the estimation quality
    bool IsGoodEstimation(); 
    // determine whether treating as keyframe
    bool IsKeyframe();

    // add matched points as observation of current keyframe
    void AddMatchedMappointsToKeyframeObservations();
    // create mappoints from new observed keypoint of current frame
    void CreateNewMappoints();
    // add new mappoints to the observedMappoints of existing keyframes in tracking map
    void AddNewMappointsObservationsForOldKeyframes();     
    // use triangulatiton to optmize the position of mappoints in tracking map
    void TriangulateMappointsInTrackingMap();

};
}

#endif // FrontEnd_H