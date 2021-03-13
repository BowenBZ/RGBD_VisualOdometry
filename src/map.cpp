/*
 * 
 */

#include "myslam/map.h"
#include "myslam/util.h"

namespace myslam
{

void Map::insertKeyFrame ( const Frame::Ptr& frame )
{
    unique_lock<mutex> lck(data_mutex_);
    keyFrames_[ frame->getID() ] = frame;
    activeKeyFrames_[ frame->getID() ] = frame;
    int remove_times = activeKeyFrames_.size() - maxActiveKeyFrameNum_;
    for (int i = 0; i < remove_times; i++) {
        removeOldKeyframe(frame);
    }
}

void Map::insertMapPoint ( const MapPoint::Ptr& map_point )
{
    unique_lock<mutex> lck(data_mutex_);
    mapPoints[map_point->getID()] = map_point;
    activeMapPoints_[map_point->getID()] = map_point;
}

void Map::removeOldKeyframe( const Frame::Ptr& curr_frame ) {

    double max_dis = 0, min_dis = 9999;
    unsigned long max_kf_id = 0, min_kf_id = 0;
    auto Twc = curr_frame->getPose().inverse();
    for (auto& kf : activeKeyFrames_) {
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
    activeKeyFrames_.erase(id_to_remove);
}

void Map::cullNonActiveMapPoints( const Frame::Ptr& currFrame ) {
    unique_lock<mutex> lck(data_mutex_);

    // remove the hardly seen and no visible points from active mappoints
    list<unsigned long> remove_id;
    for (auto& mappoint : activeMapPoints_) {
        auto mp_id = mappoint.first;
        auto mp = mappoint.second;

        // if outlider decided by backend
        if ( mp->outlier_ ) {
            remove_id.push_back(mp_id);
            continue;
        }

        // if not in current view
        if ( !currFrame->isInFrame(mp->getPosition()) ) {
            remove_id.push_back(mp_id);
            continue;
        }

        // not often matches
        float match_ratio = float(mp->matched_times_) / mp->visible_times_;
        if ( match_ratio < mapPointEraseRatio_ )
        {
            remove_id.push_back(mp_id);
            continue;
        }

        // not in good view
        double angle = getViewAngle( currFrame, mp );
        if ( angle > M_PI/6. )
        {
            remove_id.push_back(mp_id);
            continue;
        }
    }

    for(auto& id : remove_id) {
        activeMapPoints_.erase(id);
    }
}


} //namespace
