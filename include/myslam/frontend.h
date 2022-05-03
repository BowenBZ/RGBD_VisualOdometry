/*
 * Frontend for tracking
 */

#ifndef FrontEnd_H
#define FrontEnd_H

#include <opencv2/features2d/features2d.hpp>
#include <bits/stdc++.h>
#include "myslam/common_include.h"
#include "myslam/viewer.h"
#include "myslam/frame.h"
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
        INITIALIZING=-1,
        TRACKING=0,
        LOST
    };
    
    FrontEnd();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 

    void setViewer(Viewer::Ptr viewer) {viewer_ = viewer;}

    void setBackend(Backend::Ptr backend) {backend_ = backend;}

    VOState getState() { return state_;}
    
private:  
    Frame::Ptr              keyFrameRef_;   // reference keyframe
    Frame::Ptr              framePrev_;     // last frame
    Frame::Ptr              frameCurr_;     // current frame 
    VOState                 state_;         // current VO status
    cv::Ptr<cv::ORB>        orb_;           // orb detector and computer 
    vector<cv::KeyPoint>    keypointsCurr_;    // keypoints in current frame
    Mat                     descriptorsCurr_;  // descriptor in current frame 
    cv::FlannBasedMatcher   flannMatcher_;     // flann matcher
    unordered_map<MapPoint::Ptr, cv::KeyPoint>  matchedMptKptMap_;   // matched map points and keypoints
    KeyPointSet             matchedKptSet_; // set of matched keypoint
   
    SE3                     estimatedPoseCurr_; // the estimated pose of current frame 
 
    int                     num_inliers_;       // number of inlier features in pnp
    int                     accuLostFrameNums_; // number of lost times
    
    // parameters, see config/default.yaml
    float                   minDisRatio_;   // ratio for selecting good matches
    int                     maxLostFrames_; // max number of continuous lost times
    int                     min_inliers_;   // minimum inliers
    double                  keyFrameMinRot_;    // minimal rotation of two key-frames
    double                  keyFrameMinTrans_;  // minimal translation of two key-frames
    
    Viewer::Ptr             viewer_;
    Backend::Ptr            backend_;

    // inner operation 
    void extractKeyPointsAndComputeDescriptors();
    void computeDescriptors(); 
    void matchKeyPointsWithActiveMapPoints();
    void estimatePosePnP(bool addObservation); 

    // remove non-active mappoints
    // void cullNonActiveMapPoints();
    // add new mappoint to the map from the observation of a keyframe
    void addNewMapPoints();     
    // really perform the adding action
    void addNewMapPoint(int idx);     

    // add current keyframe as the new observation of old mappoints (also observed by previous keyframes)
    void addKeyframeObservationToOldMapPoints();
    
    // use triangulatiton to optmize the position of active mappoints
    // void triangulateActiveMapPoints();
    
    bool isGoodEstimation(); 
    bool isKeyFrame();
};
}

#endif // FrontEnd_H