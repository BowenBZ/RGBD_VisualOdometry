/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef FrontEnd_H
#define FrontEnd_H


#include <opencv2/features2d/features2d.hpp>
#include <bits/stdc++.h>
#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/viewer.h"
#include "myslam/frame.h"

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
    
public: // functions 
    FrontEnd();
    ~FrontEnd();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 

    void SetMap(Map::Ptr map) {map_ = map;}

    void SetViewer(Viewer::Ptr viewer) {viewer_ = viewer;}

    VOState getState() { return state_;}
    
private:  
    Map::Ptr    map_;       // map with all frames and map points
    Frame::Ptr  ref_frame_;       // reference frame 
    Frame::Ptr  curr_frame_;      // current frame 
    VOState     state_;     // current VO status
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame 
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points 
    unordered_map<MapPoint::Ptr, cv::Point2f>  match_3d_2d_pts_;   // matched 3d map points and 2d points
    unordered_set<int>      match_2dkp_index_;  // matched 2d pixels (index of keypoints_curr_)
    list<cv::Point2f>       match_2dpts_;
   
    SE3 T_c_w_estimated_;    // the estimated pose of current frame 
 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    // parameters, see config/default.yaml
    float match_ratio_;      // ratio for selecting good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double map_point_erase_ratio_; // remove map point ratio
    
    Viewer::Ptr viewer_;

    // inner operation 
    void extractKeyPoints();
    void computeDescriptors(); 
    void featureMatching();
    void poseEstimationPnP(); 

    // first key-frame, add all 3d points into map
    void initMap();
    // optimize the active mappoints in map
    void cullNonActiveMapPoints();
    // triangulate the mappoints
    void optimizeActiveMapPoints();
    
    void addMapPoints();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );
};
}

#endif // FrontEnd_H