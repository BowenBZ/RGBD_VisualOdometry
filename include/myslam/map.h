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
#include "myslam/config.h"

namespace myslam
{
class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<Map> Ptr;
    typedef unordered_map<unsigned long, MapPoint::Ptr > MappointDict;
    typedef unordered_map<unsigned long, Frame::Ptr > KeyframeDict;

    Map() {   
        mapPointEraseRatio_ = Config::get<double> ( "map_point_erase_ratio" );
    }
    
    void insertKeyFrame( const Frame::Ptr& frame );
    void insertMapPoint( const MapPoint::Ptr& map_point );

    void removeActiveMapPoint ( const unsigned long& id ) {
        unique_lock<mutex> lck(data_mutex_);
        activeMapPoints_.erase(id);
    }

    // cull the hardly seen and no visible points of current frame from active mappoints
    void cullNonActiveMapPoints( const Frame::Ptr& currFrame );

    void updateMappointEraseRatio() {
        mapPointEraseRatio_ = (activeMapPoints_.size() > 1000 ) ? 
                                    mapPointEraseRatio_ + 0.05 :
                                    0.1;
    }

    KeyframeDict getAllKeyFrames() {
        unique_lock<mutex> lck(data_mutex_);
        return keyFrames_;
    }
    MappointDict getAllMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return mapPoints;
    }

    MappointDict getActiveMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        return activeMapPoints_;
    }

    void resetActiveMappoints() {
        unique_lock<mutex> lck(data_mutex_);
        activeMapPoints_ = mapPoints;
    }

private:
    mutex data_mutex_;

    MappointDict  mapPoints;        // all mappoints
    KeyframeDict  keyFrames_;         // all key-frames

    MappointDict  activeMapPoints_;        // active mappoints, used for feature matching in frontend

    float mapPointEraseRatio_;

    void removeOldKeyframe( const Frame::Ptr& curr_frame);

};

} //namespace

#endif // MAP_H
