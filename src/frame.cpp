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
#include "myslam/mapmanager.h"

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

void Frame::removeObservedMapPoint(const shared_ptr<MapPoint> mpt) {
    unique_lock<mutex> lck(observationMutex_);

    bool flag = false;
    for (auto iter = observedMapPoints_.begin(); iter != observedMapPoints_.end(); iter++) {
        if (iter->lock() == mpt) {
            observedMapPoints_.erase(iter);
            flag = true;
            break;
        }
    }

    if (flag) {
        for (auto& pair: mpt->getKeyFrameObservationsMap()) {
            auto otherKF = MapManager::GetInstance().GetKeyframe(pair.first);

            if ( otherKF == nullptr || otherKF->getId() == this->id_ ) {
                continue;
            }

            this->decreaseConnectedKeyFrameWeightByOne(otherKF->getId());
            otherKF->decreaseConnectedKeyFrameWeightByOne(this->id_);
        }
    }

    // if all the observations has been removed, consider this keyframe as outlier?
}

void Frame::decreaseConnectedKeyFrameWeightByOne(const unsigned long id) {
    if (connectedKeyFrameIdToWeight_.count(id)) {
        connectedKeyFrameIdToWeight_[id]--;
        if (connectedKeyFrameIdToWeight_[id] < 15 
            && connectedKeyFrameIdToWeight_.size() >= 2) {
                connectedKeyFrameIdToWeight_.erase(id);
            }
    }
}

void Frame::updateConnectedKeyFrames() {
    unique_lock<mutex> lck(connectedMutex_);

    // Calcualte the co-visibliblity keyframes' weight
    ConnectedKeyFrameIdToWeight connectedKeyFrameCandidates;
    for(auto& mapPoint : observedMapPoints_) {
        if (mapPoint.expired() || mapPoint.lock()->outlier_) {
            continue;
        }

        for(auto& keyFrameMap : (mapPoint.lock())->getKeyFrameObservationsMap()) {

            auto keyFrame = MapManager::GetInstance().GetKeyframe(keyFrameMap.first);

            if ( keyFrame == nullptr || keyFrame->getId() == id_) {
                continue;
            }

            connectedKeyFrameCandidates[keyFrame->getId()]++;
        }
    }
       
    // Filter the connections whose weight larger than 15
    connectedKeyFrameIdToWeight_.clear();
    int maxWeight = 0;
    unsigned long maxWeightConnectedKeyFrameId;

    for(auto& pair : connectedKeyFrameCandidates) {
        auto connectedKeyFrameId = pair.first;
        auto connectedMapPointsCnt = pair.second;

        if(connectedMapPointsCnt >= 15) {
            connectedKeyFrameIdToWeight_[connectedKeyFrameId] = connectedMapPointsCnt;
            MapManager::GetInstance().GetKeyframe(connectedKeyFrameId)->addConnectedKeyFrame(this->id_, connectedMapPointsCnt);
        }

        if(connectedMapPointsCnt > maxWeight) {
            maxWeightConnectedKeyFrameId = connectedKeyFrameId;
            maxWeight = connectedMapPointsCnt;
        }
    }

    // In case there is no weight larger than 15
    if(connectedKeyFrameIdToWeight_.empty() && maxWeight != 0) {
        connectedKeyFrameIdToWeight_[maxWeightConnectedKeyFrameId] = maxWeight;
        MapManager::GetInstance().GetKeyframe(maxWeightConnectedKeyFrameId)->addConnectedKeyFrame(this->id_, maxWeight);
    }

    // cout << "Current Keyframe Id: " << this->id_ << endl;
    // cout << "Connected Keyframe counts: " << connectedKeyFrameIdToWeight_.size() << endl;
    // for(auto& pair: connectedKeyFrameIdToWeight_) {
    //     cout << "Id: " << pair.first << " Weight: " << pair.second << endl;
    // }    
}


}
