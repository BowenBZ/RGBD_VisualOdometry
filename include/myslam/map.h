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

#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam
{
class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<Map> Ptr;
    typedef unordered_map<unsigned long, MapPoint::Ptr > MappointDict;
    typedef unordered_map<unsigned long, Frame::Ptr > KeyframeDict;

    Map() {}
    
    void insertKeyFrame( Frame::Ptr frame );
    void insertMapPoint( MapPoint::Ptr map_point );

    KeyframeDict getAllKeyFrames() {
        unique_lock<mutex> lck(data_mutex_);
        return keyframes_;
    }
    MappointDict getAllMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return map_points_;
    }

    KeyframeDict getActiveKeyFrames() {
        unique_lock<mutex> lck(data_mutex_);
        return active_keyframes_;

    }
    MappointDict getActiveMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return active_map_points_;
    }

private:
    mutex data_mutex_;

    MappointDict  map_points_;        // all landmarks
    KeyframeDict  keyframes_;         // all key-frames

    MappointDict  active_map_points_;        // all landmarks
    KeyframeDict  active_keyframes_;         // all key-frames
};

} //namespace

#endif // MAP_H
