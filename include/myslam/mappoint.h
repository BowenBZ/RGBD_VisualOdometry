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
    unsigned long      id_; // ID
    static unsigned long factory_id_;    // factory id
    bool        good_;      // wheter a good point 
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 

    list<Frame*>    observed_frames_;   // frames that can observe this point 
    vector<cv::Point2f>    observed_pixel_pos_;    // pixel pos observed by the frames

    int         visible_times_;     // being visible in current frame 
    int         matched_times_;     // being an inliner in pose estimation
    
    MapPoint();
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, 
        const Vector3d& norm, 
        cv::Point2f pixel_pos,
        Frame* frame=nullptr, 
        const Mat& descriptor=Mat()
    );
    
    inline cv::Point3f getPositionCV() const {
        return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
    }

    // factory function
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint( 
        const Vector3d& pos_world, 
        const Vector3d& norm_,
        const cv::Point2f& pixel_pos,
        const Mat& descriptor,
        Frame* frame );
};
}

#endif // MAPPOINT_H
