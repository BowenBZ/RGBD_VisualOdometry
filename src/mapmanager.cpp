/*
 * MapManager which maintains the mappoints and keyframes with <id, shared_ptr> pair. Other components should only maintain the index of respective resources
 * 
 * There is only 1 MapManager instance, other components need to use MapManager.GetInstance() to get the instance
 * 
 * MapManager will use the dataMutex_ when modifying or returning the mappoints or keyframes.
 */

#include "myslam/mapmanager.h"

namespace myslam
{

// void MapManager::cullNonActiveMapPoints( const Frame::Ptr& currFrame ) {
//     unique_lock<mutex> lck(dataMutex_);

//     // remove the hardly seen and no visible points from active mappoints
//     list<unsigned long> remove_id;
//     for (auto& mappoint : activeMapPoints_) {
//         auto mp_id = mappoint.first;
//         auto mp = mappoint.second;

//         // if outlider decided by backend
//         if ( mp->outlier_ ) {
//             remove_id.push_back(mp_id);
//             continue;
//         }

//         // if not in current view
//         if ( !currFrame->IsInFrame(mp->GetPosition()) ) {
//             remove_id.push_back(mp_id);
//             continue;
//         }

//         // not often matches
//         float match_ratio = float(mp->matchedTimes_) / mp->visibleTimes_;
//         if ( match_ratio < mapPointEraseRatio_ )
//         {
//             remove_id.push_back(mp_id);
//             continue;
//         }

//         // not in good view
//         // TODO: update the norm_ direction of mp
//         // Vector3d direction = mp->GetPosition() - currFrame->GetCamCenter();
//         // direction.normalize();
//         // double angle = acos( direction.transpose() * mp->norm_ );
//         // if ( angle > M_PI/6. )
//         // {
//         //     remove_id.push_back(mp_id);
//         //     continue;
//         // }
//     }

//     for(auto& id : remove_id) {
//         activeMapPoints_.erase(id);
//     }
// }

MapManager::MappointDict MapManager::GetMappointsAroundKeyframe( const Frame::Ptr& keyframe ) {
    unique_lock<mutex> lck(dataMutex_);

    auto localKeyframes = keyframe->GetCovisibleKeyframes();
    // Add this keyFrame to the connected keyframe map
    localKeyframes[keyframe->GetId()] = 0;

    unordered_map<size_t, Mappoint::Ptr> localMappointsDict;

    // Find all mappoints observed by connected keyframes
    for(auto& idToWeight: localKeyframes) {
        if (!keyframesDict_.count(idToWeight.first)) {
            continue;
        }
        auto localKeyframe = keyframesDict_[idToWeight.first];

        for(auto& mappointId: localKeyframe -> GetObservedMappointIds()) {
            if (!mappointsDict_.count(mappointId) || mappointsDict_[mappointId] -> outlier_) {
                continue;
            }

            localMappointsDict[mappointId] = mappointsDict_[mappointId];
        }
    }

    return localMappointsDict;
}

inline double getViewAngle ( const Frame::Ptr& frame, const Mappoint::Ptr& point )
{
    Vector3d n = point->GetPosition() - frame->GetCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

} //namespace
