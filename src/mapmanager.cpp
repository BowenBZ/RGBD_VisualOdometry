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

} //namespace
