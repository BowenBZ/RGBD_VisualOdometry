/*
 * MapManager which maintains the mappoints and keyframes with <id, shared_ptr> pair. Other components should only maintain the index of respective resources
 */

#include "myslam/private/mapmanager.h"

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
        const auto& kf = keyframesDict_[kfId];

        for(auto& [mptId, _]: kf->GetAllObservingMptIdToKptIdx()) {
            assert(mappointsDict_.count(mptId));

            const auto& mpt = mappointsDict_[mptId];
            // TODO: don't return outlier or non-optimized
            // if (mpt->outlier_ || !mpt->optimized_) {
            //     continue;
            // }

            mptIdToMpt[mptId] = mpt;
        }
    }
}

} //namespace
