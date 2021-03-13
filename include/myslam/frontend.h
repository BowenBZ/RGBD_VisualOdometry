/*
 * Frontend for tracking
 */

#ifndef FrontEnd_H
#define FrontEnd_H

#include <opencv2/features2d/features2d.hpp>
#include <bits/stdc++.h>
#include "myslam/common_include.h"
#include "myslam/map.h"
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

    void setMap(Map::Ptr map) {map_ = map;}

    void setViewer(Viewer::Ptr viewer) {viewer_ = viewer;}

    void setBackend(Backend::Ptr backend) {backend_ = backend;}

    VOState getState() { return state_;}
    
private:  
    Map::Ptr    map_;       // map with all frames and map points
    Frame::Ptr  frameRef_;       // reference frame 
    Frame::Ptr  frameCurr_;      // current frame 
    VOState     state_;     // current VO status
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    vector<cv::KeyPoint>    keypointsCurr_;    // keypoints in current frame
    Mat                     descriptorsCurr_;  // descriptor in current frame 
    cv::FlannBasedMatcher   flannMatcher_;     // flann matcher
    unordered_map<MapPoint::Ptr, cv::KeyPoint>  matchedMptKptMap_;   // matched map points and keypoints
    unordered_set<cv::KeyPoint, KeyPointHash, KeyPointsComparision>  matchedKptSet_; // set of matched keypoint
   
    SE3 estimatedPoseCurr_;    // the estimated pose of current frame 
 
    int num_inliers_;        // number of inlier features in pnp
    int num_lost_;           // number of lost times
    
    // parameters, see config/default.yaml
    float match_ratio_;      // ratio for selecting good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    
    // inner operation 
    void extractKeyPointsAndComputeDescriptors();
    void computeDescriptors(); 
    void matchKeyPointsWithActiveMapPoints();
    void estimatePosePnP(); 
    // for first key-frame, add all 3d points into map
    void initMap();
    // remove non-active mappoints from active map, add more mappoints if the the map is small
    void updateActiveMapPointsMap();
    // use backend or triangulatiton to optmize the position of mappoints (and pose of frames when use backend)
    void optimizeActiveMapPointsPosition();
    
    void addNewMapPoints();
    bool isGoodEstimation(); 
    bool isKeyFrame();

    Viewer::Ptr viewer_;

    Backend::Ptr backend_;

};
}

#endif // FrontEnd_H