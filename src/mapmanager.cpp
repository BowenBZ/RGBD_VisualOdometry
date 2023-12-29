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

MapManager::MappointIdToPtr MapManager::GetMappointsAroundKeyframe( const Frame::Ptr& keyframe ) {
    unique_lock<mutex> lck(dataMutex_);

    auto covisibleKeyframeIds = keyframe->GetCovisibleKeyframes();
    // Add this keyFrame to the covisible keyframe map
    covisibleKeyframeIds.insert(keyframe->GetId());

    unordered_map<size_t, Mappoint::Ptr> localMappointsDict;

    // Find all mappoints observed by connected keyframes
    for(auto& keyframeId: covisibleKeyframeIds) {
        assert(keyframesDict_.count(keyframeId));
        auto& localKeyframe = keyframesDict_[keyframeId];

        for(auto& mappointId: localKeyframe -> GetObservedMappointIds()) {
            if (!mappointsDict_.count(mappointId) || mappointsDict_[mappointId] -> outlier_) {
                continue;
            }

            localMappointsDict[mappointId] = mappointsDict_[mappointId];
        }
    }

    return localMappointsDict;
}

} //namespace
