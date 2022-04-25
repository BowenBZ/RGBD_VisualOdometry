/*
 * 
 */

#include "myslam/map.h"

namespace myslam
{

void Map::insertKeyFrame ( const Frame::Ptr& frame )
{
    unique_lock<mutex> lck(data_mutex_);
    keyFrames_[ frame->getId() ] = frame;
}

void Map::insertMapPoint ( const MapPoint::Ptr& map_point )
{
    unique_lock<mutex> lck(data_mutex_);
    mapPoints[map_point->getId()] = map_point;
    activeMapPoints_[map_point->getId()] = map_point;
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
        // TODO: update the norm_ direction of mp
        // Vector3d direction = mp->getPosition() - currFrame->getCamCenter();
        // direction.normalize();
        // double angle = acos( direction.transpose() * mp->norm_ );
        // if ( angle > M_PI/6. )
        // {
        //     remove_id.push_back(mp_id);
        //     continue;
        // }
    }

    for(auto& id : remove_id) {
        activeMapPoints_.erase(id);
    }
}

Map::MappointDict Map::getLocalMappoints( const Frame::Ptr& keyFrame ) {
    unique_lock<mutex> lck(data_mutex_);

    auto connectedKeyFrames = keyFrame->getConnectedKeyFrames();
    // Add this keyFrame to the connected keyframe map
    connectedKeyFrames[keyFrame->getId()] = 0;

    unordered_map<uint64_t, MapPoint::Ptr> mapPointMap;

    // Find all mappoints observed by connected keyframes
    for(auto& pair: connectedKeyFrames) {
    
        auto keyFrameId = pair.first;
        auto keyFrame = getKeyFrame(keyFrameId);

        if (keyFrame == nullptr) {
            continue;
        }

        for(auto& mapPointPtr: keyFrame -> getObservedMapPoints()) {
            if ( mapPointPtr.expired() ) {
                continue;
            }
            auto mapPoint = mapPointPtr.lock();
            if(mapPoint->outlier_) {
                continue;
            }

            mapPointMap[mapPoint -> getId()] = mapPoint;
        }
    }

    return mapPointMap;
}

inline double getViewAngle ( const Frame::Ptr& frame, const MapPoint::Ptr& point )
{
    Vector3d n = point->getPosition() - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

} //namespace
