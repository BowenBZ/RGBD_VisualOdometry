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

#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/util.h"

namespace myslam 
{
    
// forward declare 
class MapPoint;

class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<Frame> Ptr;
    typedef unordered_map<Frame::Ptr, int> ConnectedKeyFrameMapType;
    double                         time_stamp_; // when it is recorded
    Camera::Ptr                    camera_;     // Pinhole RGBD Camera model 
    Mat                            color_, depth_; // color and depth image 
    
    Frame();
    Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
    ~Frame();
    
    // factory function
    static Frame::Ptr createFrame(); 
    
    // find the depth in depth map
    double findDepth( const cv::KeyPoint& kp );
    
    // Get Camera Center
    Vector3d getCamCenter() const;
    
    // check if a point is in this frame 
    bool isInFrame( const Vector3d& pt_world );

    SE3 getPose() {
        unique_lock<mutex> lck(poseMutex_);
        return T_c_w_;
    }

    SE3 setPose(const SE3& pose) {
        unique_lock<mutex> lck(poseMutex_);
        T_c_w_ = pose;
    }

    unsigned long getID() { return id_; }

    // Update the co-visible key-frames when this frame is a key-frame 
    void updateConnectedKeyFrames();

    // Add the connection of another frame with weight to current frame
    void addConnectedKeyFrame(const Frame::Ptr& frame, const int& weight) {
        unique_lock<mutex> lck(connecedMutex_);
        connectedKeyFramesCounter_[frame] = weight;
    }

    ConnectedKeyFrameMapType getConnectedKeyFrames() {
        unique_lock<mutex> lck(connecedMutex_);
        return connectedKeyFramesCounter_;
    }

    void addObservedMapPoint(const weak_ptr<MapPoint>& mpt) {
        unique_lock<mutex> lck(connecedMutex_);
        observedMapPoints_.push_back(mpt);
    }

    list<weak_ptr<MapPoint>> getObservedMapPoints() {
        unique_lock<mutex> lck(connecedMutex_);
        return observedMapPoints_;
    }

private: 
    static unsigned long factoryId_;
    unsigned long               id_;         // id of this frame

    mutex poseMutex_;
    SE3                         T_c_w_;      // transform from world to camera

    mutex connecedMutex_;
    // Connected keyframes (has same observed mappoints >= 15) and the number of mappoints
    ConnectedKeyFrameMapType connectedKeyFramesCounter_;

    list<weak_ptr<MapPoint>> observedMapPoints_;
};

}

#endif // FRAME_H
