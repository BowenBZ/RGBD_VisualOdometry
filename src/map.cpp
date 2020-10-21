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

#include "myslam/map.h"

namespace myslam
{

void Map::insertKeyFrame ( Frame::Ptr frame )
{
    unique_lock<mutex> lck(data_mutex_);
    keyframes_[ frame->id_ ] = frame;
    active_keyframes_[ frame->id_ ] = frame;
    
}

void Map::insertMapPoint ( MapPoint::Ptr map_point )
{
    unique_lock<mutex> lck(data_mutex_);
    map_points_[map_point->id_] = map_point;
    active_map_points_[map_point->id_] = map_point;
}


} //namespace
