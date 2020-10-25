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

void Map::insertKeyFrame ( const Frame::Ptr& frame )
{
    unique_lock<mutex> lck(data_mutex_);
    keyframes_[ frame->getID() ] = frame;
    active_keyframes_[ frame->getID() ] = frame;
    int remove_times = active_keyframes_.size() - active_keyframes_num_;
    for (int i = 0; i < remove_times; i++) {
        removeOldKeyframe(frame);
    }
}

void Map::insertMapPoint ( const MapPoint::Ptr& map_point )
{
    unique_lock<mutex> lck(data_mutex_);
    map_points_[map_point->getID()] = map_point;
    active_map_points_[map_point->getID()] = map_point;
}

void Map::removeOldKeyframe( const Frame::Ptr& curr_frame ) {

    double max_dis = 0, min_dis = 9999;
    unsigned long max_kf_id = 0, min_kf_id = 0;
    auto Twc = curr_frame->getPose().inverse();
    for (auto& kf : active_keyframes_) {
        if (kf.first == curr_frame->getID()) 
            continue;

        auto dis = (kf.second->getPose() * Twc).log().norm();
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;
    unsigned long id_to_remove = (min_dis < min_dis_th) ? min_kf_id : max_kf_id;
    active_keyframes_.erase(id_to_remove);
}


} //namespace
