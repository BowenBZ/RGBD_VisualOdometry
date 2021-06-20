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

MapPoint::MapPoint ( 
    long unsigned int id, 
    const Vector3d& position, 
    const Vector3d& norm, 
    const Mat& descriptor,
    const weak_ptr<Frame>& observedByKeyFrame,
    const cv::Point2f& pixelPos)
: id_(id), pos_(position), norm_(norm), triangulated_(false), visibleTimes_(1), matchedTimes_(1), descriptor_(descriptor), outlier_(false)
{
    addKeyFrameObservation(observedByKeyFrame, pixelPos);
}

MapPoint::Ptr MapPoint::createMapPoint ( 
    const Vector3d& posWorld, 
    const Vector3d& norm,
    const Mat& descriptor,
    const weak_ptr<Frame>& observedByKeyFrame,
    const cv::Point2f& pixelPos)
{
    return MapPoint::Ptr( 
        new MapPoint( factoryId_++, posWorld, norm, descriptor, observedByKeyFrame, pixelPos)
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

    // if all the observations has been removed
    if(observedKeyFrameMap_.size() == 0) {
        outlier_ = true;
    }
}


} // namespace