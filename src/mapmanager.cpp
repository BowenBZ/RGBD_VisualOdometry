/*
 * MapManager which maintains the mappoints and keyframes with <id, shared_ptr> pair. Other components should only maintain the index of respective resources
 * 

 * 
 * MapManager will use the dataMutex_ when modifying or returning the mappoints or keyframes.
 */

#include "myslam/mapmanager.h"

namespace myslam
{

void MapManager::GetMappointsNearKeyframe(const Frame::Ptr& keyframe, MappointIdToPtr& mptIdToMpt) {
    mptIdToMpt.clear();

    list<size_t> allCovisibleKfIds;
    keyframe->GetAllCovisibleKfIds(allCovisibleKfIds);
    // Add current keyFrame to the covisible keyframe map
    allCovisibleKfIds.push_back(keyframe->GetId());

    // find all mappoints observed by keyframes above
    for(const auto& kfId: allCovisibleKfIds) {
        assert(keyframesDict_.count(kfId));
        auto& kf = keyframesDict_[kfId];

        list<size_t> observingMptIds;
        kf->GetObservingMappointIds(observingMptIds);
        for(auto& mptId: observingMptIds) {
            assert(mappointsDict_.count(mptId));

            auto& mpt = mappointsDict_[mptId];
            if (mpt->outlier_ || !mpt->optimized_) {
                continue;
            }

            mptIdToMpt[mptId] = mpt;
        }
    }
}

void MapManager::ReplaceMappoint(size_t oldMptId, size_t newMptId) {

    oldMptIdNewMptIdMap_[oldMptId] = newMptId;
    
    assert(mappointsDict_.count(oldMptId));
    assert(mappointsDict_.count(newMptId));
    assert(oldMptId != newMptId);

    auto& oldMpt = mappointsDict_[oldMptId];
    auto& newMpt = mappointsDict_[newMptId];

    unordered_map<size_t, size_t> observedByKfIdToKptIdx;
    oldMpt->GetObservedByKeyframesMap(observedByKfIdToKptIdx);
    for(auto& [kfId, kptIdx]: observedByKfIdToKptIdx) {
        assert(keyframesDict_.count(kfId));
        auto& kf = keyframesDict_[kfId];
 
        kf->RemoveObservingMappoint(oldMptId);
        // If the kf already observes the newMpt, which could happen when several old mpts need to be replaced by the same new mpt. This means those points should be merged together, so just remove one observation
        if (kf->IsObservingMappoint(newMptId)) {
            kf->RemoveObservingMappoint(newMptId);
        }
        kf->AddObservingMappoint(newMpt, kptIdx);
    }
    mappointsDict_.erase(oldMptId);
}

Mappoint::Ptr MapManager::GetPotentialReplacedMappoint(const size_t oldMptId) {

    if (oldMptIdNewMptIdMap_.count(oldMptId)) {
        size_t newMptId = oldMptIdNewMptIdMap_[oldMptId];
        while (oldMptIdNewMptIdMap_.count(newMptId)) {
            newMptId = oldMptIdNewMptIdMap_[newMptId];
        }
        return mappointsDict_[newMptId];
    } else {
        return mappointsDict_[oldMptId];
    }
}

} //namespace
