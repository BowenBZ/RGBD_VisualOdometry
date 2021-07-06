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

namespace myslam
{
class Frame;

class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<MapPoint> Ptr;
    typedef list<pair<unsigned long, cv::Point2f>> ObservedKFtoPixelPos;

    // Usages
    // 1. whether match with new mappoints in front end
    // 2. whether will be triangulated
    // 3. whether add new observation keyframe 
    // 4. whether update in co-visibility graph
    // 5. whether add into backend
    bool        outlier_;               // whether this is an outlider

    bool        optimized_;             // whether is optimized by backend

    bool        triangulated_;          // whether have been triangulated
    Vector3d    norm_;                  // Normal of viewing direction 

    Mat         descriptor_;            // Descriptor for matching 
    int         visibleTimes_;          // times should in the view of current frame, but maybe cannot be matched 
    int         matchedTimes_;          // times of being an inliner in frontend P3P result
    
    // factory function
    static MapPoint::Ptr createMapPoint( 
        const Vector3d posWorld, 
        const Vector3d norm,
        const Mat descriptor,
        const unsigned long observedKeyFrameId,
        const cv::Point2f pixelPos);

    Vector3d getPosition() {
        unique_lock<mutex> lock(posMutex_);
        return pos_;
    }

    void setPosition(const Vector3d& pos) {
        unique_lock<mutex> lock(posMutex_);
        pos_ = pos;
    }

    unsigned long getId() { return id_; }

    void addKeyFrameObservation(const unsigned long keyFrameId, const cv::Point2f& pixel_pos) {
        unique_lock<mutex> lock(observationMutex_);
        observedKeyFrameMap_.push_back(make_pair(keyFrameId, pixel_pos));
    }

    ObservedKFtoPixelPos getKeyFrameObservationsMap() {
        unique_lock<mutex> lock(observationMutex_);
        return observedKeyFrameMap_;
    }

    void removeKeyFrameObservation(const unsigned long keyFrameId);

private:
    static unsigned long factoryId_;    // factory id
    unsigned long      id_; // ID

    mutex posMutex_;
    Vector3d    pos_;       // Position in world

    mutex observationMutex_;
    ObservedKFtoPixelPos observedKeyFrameMap_;

    // mappoint can only be created by factory
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, 
        const Vector3d& norm, 
        const Mat& descriptor,
        const unsigned long observedKeyFrameId,
        const cv::Point2f& pixelPos);
};

} // namespace

#endif // MAPPOINT_H
