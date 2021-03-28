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

#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam
{
Frame::Frame()
: id_(-1), time_stamp_(-1), camera_(nullptr)
{

}

Frame::Frame ( long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth )
: id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
{

}

Frame::~Frame()
{

}

unsigned long Frame::factoryId_ = 0;

Frame::Ptr Frame::createFrame()
{
    return Frame::Ptr( new Frame(factoryId_++) );
}

double Frame::findDepth ( const cv::KeyPoint& kp )
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return double(d)/camera_->depth_scale_;
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return double(d)/camera_->depth_scale_;
            }
        }
    }
    return -1.0;
}


Vector3d Frame::getCamCenter() const
{
    return T_c_w_.inverse().translation();
}

bool Frame::isInFrame ( const Vector3d& pt_world )
{
    Vector3d p_cam = camera_->world2camera( pt_world, T_c_w_ );
    if ( p_cam(2, 0) < 0 ) {
        return false;
    } 
    Vector2d pixel = camera_->camera2pixel ( p_cam );
    return pixel(0,0)>0 && pixel(1,0)>0 
        && pixel(0,0)<color_.cols 
        && pixel(1,0)<color_.rows;
}

void Frame::updateConnectedKeyFrames() {
    unique_lock<mutex> lck(connecedMutex_);

    // Calcualte the co-visibliblity keyframes' weight
    ConnectedKeyFrameMapType connectedKeyFrameCandidates;
    for(auto& mapPoint : observedMapPoints_) {
        if (mapPoint.expired()) {
            continue;
        }

        for(auto& keyFrameMap : (mapPoint.lock())->getKeyFrameObservationsMap()) {

            auto keyFrame = keyFrameMap.first;

            if (!keyFrame.expired() && keyFrame.lock()->getID() != id_) {
                connectedKeyFrameCandidates[keyFrame]++;
            }
        }
    }
       
    // Filter the connections whose weight larger than 15
    connectedKeyFramesCounter_.clear();
    int maxCount = 0;
    weak_ptr<Frame> maxCountKeyFrame;

    for(auto& connectedKeyFrame : connectedKeyFrameCandidates) {
        if(connectedKeyFrame.first.lock() && connectedKeyFrame.second >= 15) {
            connectedKeyFramesCounter_[connectedKeyFrame.first] = connectedKeyFrame.second;
            (connectedKeyFrame.first.lock())->addConnectedKeyFrame(Ptr(this), connectedKeyFrame.second);
        }
        if(connectedKeyFrame.second > maxCount) {
            maxCountKeyFrame = connectedKeyFrame.first;
            maxCount = connectedKeyFrame.second;
        }
    }

    // In case there is no weight larger than 15
    if(connectedKeyFramesCounter_.empty()) {
        connectedKeyFramesCounter_[maxCountKeyFrame] = maxCount;
        (maxCountKeyFrame.lock())->addConnectedKeyFrame(Ptr(this), maxCount);
    }    
}


}
