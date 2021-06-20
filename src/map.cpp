/*
 * 
 */

#include "myslam/map.h"

namespace myslam
{

void Map::insertKeyFrame ( const Frame::Ptr& frame )
{
    unique_lock<mutex> lck(data_mutex_);
    keyFrames_[ frame->getID() ] = frame;
}

void Map::insertMapPoint ( const MapPoint::Ptr& map_point )
{
    unique_lock<mutex> lck(data_mutex_);
    mapPoints[map_point->getID()] = map_point;
    activeMapPoints_[map_point->getID()] = map_point;
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
        float match_ratio = float(mp->matchedTimes_) / mp->visibleTimes_;
        if ( match_ratio < mapPointEraseRatio_ )
        {
            remove_id.push_back(mp_id);
            continue;
        }

        // not in good view
        Vector3d direction = mp->getPosition() - currFrame->getCamCenter();
        direction.normalize();
        double angle = acos( direction.transpose() * mp->norm_ );
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

inline double getViewAngle ( const Frame::Ptr& frame, const MapPoint::Ptr& point )
{
    Vector3d n = point->getPosition() - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

} //namespace
