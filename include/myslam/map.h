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

    Map() {   }
    
    void insertKeyFrame( const Frame::Ptr& frame );
    void insertMapPoint( const MapPoint::Ptr& map_point );

    void removeActiveMapPoint ( const unsigned long& id ) {
        unique_lock<mutex> lck(data_mutex_);
        active_map_points_.erase(id);
    }

    void removeActiveMapPoints ( const list<unsigned long>& ids ) {
        unique_lock<mutex> lck(data_mutex_);
        for(auto& id : ids) {
            active_map_points_.erase(id);
        }
    }

    KeyframeDict getAllKeyFrames() {
        unique_lock<mutex> lck(data_mutex_);
        return keyframes_;
    }
    MappointDict getAllMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return map_points_;
    }

    MappointDict getActiveMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return active_map_points_;
    }

    void resetActiveMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        active_map_points_ = map_points_;
    }

private:
    mutex data_mutex_;

    MappointDict  map_points_;        // all mappoints
    KeyframeDict  keyframes_;         // all key-frames

    MappointDict  active_map_points_;        // active mappoints, used for feature matching in frontend

};

} //namespace

#endif // MAP_H
