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

#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

MapPoint::MapPoint()
: id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), triangulated_(false), visible_times_(0), matched_times_(0)
{

}


MapPoint::MapPoint ( long unsigned int id, const Vector3d& position, const Vector3d& norm, 
                     const cv::Point2f& pixel_pos, const weak_ptr<Frame>& frame, const Mat& descriptor )
: id_(id), pos_(position), norm_(norm), triangulated_(false), visible_times_(1), matched_times_(1), descriptor_(descriptor), outlier_(false)
{
    addFrameObservation(frame, pixel_pos);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, Vector3d(0,0,0), Vector3d(0,0,0), cv::Point2f(0, 0) )
    );
}

MapPoint::Ptr MapPoint::createMapPoint ( 
    const Vector3d& pos_world, 
    const Vector3d& norm,
    const cv::Point2f& pixel_pos, 
    const Mat& descriptor, 
    const Frame::Ptr& frame )
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, pos_world, norm, pixel_pos, frame, descriptor )
    );
}

unsigned long MapPoint::factory_id_ = 0;


void MapPoint::removeFrameObservation(const Frame::Ptr& frame) {
    unique_lock<mutex> lck(observation_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end(); iter++) {
        if (iter->first.lock() == frame) {
            observations_.erase(iter);
            break;
        }
    }

    // if all the observations has been removed, or only left the first frame (maybe not keyframe)
    if(observations_.size() <= 1)
        outlier_ = true;
}


} // namespace