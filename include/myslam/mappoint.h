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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam
{
    
class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<MapPoint> Ptr;
    typedef list<pair<weak_ptr<Frame>, cv::Point2f>> ObservationType;
    bool        triangulated_;          // whether have been triangulated
    bool        outlier_;               // whether this is an outlider
    Vector3d    norm_;                  // Normal of viewing direction 

    Mat         descriptor_;            // Descriptor for matching 
    int         visible_times_;         // times should in the view of current frame, but maybe cannot be matched 
    int         matched_times_;         // being an inliner in pose estimation
    
    MapPoint();
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, 
        const Vector3d& norm, 
        const cv::Point2f& pixel_pos,
        const weak_ptr<Frame>& frame=weak_ptr<Frame>(), 
        const Mat& descriptor=Mat()
    );
    
    Vector3d getPosition() {
        unique_lock<mutex> lock(pos_mutex_);
        return pos_;
    }

    cv::Point3f getPositionCV() {
        unique_lock<mutex> lock(pos_mutex_);
        return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
    }

    void setPosition(const Vector3d& pos) {
        unique_lock<mutex> lock(pos_mutex_);
        pos_ = pos;
    }

    unsigned long getID() { return id_; }

    // factory function
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint( 
        const Vector3d& pos_world, 
        const Vector3d& norm_,
        const cv::Point2f& pixel_pos,
        const Mat& descriptor,
        const Frame::Ptr& frame );

    void addFrameObservation(const weak_ptr<Frame>& frame, const cv::Point2f& pixel_pos) {
        unique_lock<mutex> lock(observation_mutex_);
        observations_.push_back(make_pair(frame, pixel_pos));
    }

    ObservationType getFrameObservations() {
        unique_lock<mutex> lock(observation_mutex_);
        return observations_;
    }

    void removeFrameObservation(const Frame::Ptr& frame);

private:
    static unsigned long factory_id_;    // factory id
    unsigned long      id_; // ID

    mutex pos_mutex_;
    Vector3d    pos_;       // Position in world

    mutex observation_mutex_;
    ObservationType observations_;
};

} // namespace

#endif // MAPPOINT_H
