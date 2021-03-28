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
: id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), triangulated_(false), visibleTimes_(0), matchedTimes_(0)
{

}


MapPoint::MapPoint ( long unsigned int id, const Vector3d& position, const Vector3d& norm, 
                     const cv::Point2f& pixel_pos, const weak_ptr<Frame>& frame, const Mat& descriptor )
: id_(id), pos_(position), norm_(norm), triangulated_(false), visibleTimes_(1), matchedTimes_(1), descriptor_(descriptor), outlier_(false)
{
    addKeyFrameObservation(frame, pixel_pos);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return MapPoint::Ptr( 
        new MapPoint( factoryId_++, Vector3d(0,0,0), Vector3d(0,0,0), cv::Point2f(0, 0) )
    );
}

MapPoint::Ptr MapPoint::createMapPoint ( 
    const Vector3d& posWorld, 
    const Vector3d& norm,
    const cv::Point2f& pixel_pos, 
    const Mat& descriptor, 
    const shared_ptr<Frame>& frame )
{
    return MapPoint::Ptr( 
        new MapPoint( factoryId_++, posWorld, norm, pixel_pos, frame, descriptor )
    );
}

unsigned long MapPoint::factoryId_ = 0;


void MapPoint::removeKeyFrameObservation(const shared_ptr<Frame>& frame) {
    unique_lock<mutex> lck(observationMutex_);
    for (auto iter = observedKeyFrameMap_.begin(); iter != observedKeyFrameMap_.end(); iter++) {
        if (iter->first.lock() == frame) {
            observedKeyFrameMap_.erase(iter);
            break;
        }
    }

    // if all the observations has been removed, or only left the first frame (maybe not keyframe)
    if(observedKeyFrameMap_.size() <= 1)
        outlier_ = true;
}


} // namespace