/*
 * Maintains the keyframes and mappoints created from keyframes
 * 
 * There is only 1 MapManager instance, other components need to use MapManager.Instance() to get the instance
 * 
 * Provides following functions
 * 1. replace old mappoint with a new mappoint
 * 2. get nearby mappoints for a keyframe from its covisible graphs (mappoints observed by this keyframe and its active covisible keyframes)
 */

#ifndef MAPMANAGER_H
#define MAPMANAGER_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"

namespace myslam
{
class MapManager
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<MapManager> Ptr;
    typedef unordered_map<size_t, Mappoint::Ptr > MappointIdToPtr;
    typedef unordered_map<size_t, Frame::Ptr > KeyframeIdToPtr;

    static MapManager& Instance() {
        static MapManager map_;
        return map_;
    }
    
    void AddKeyframe(const Frame::Ptr& frame) {
        unique_lock<mutex> lck(mutex_);
        keyframesDict_[ frame->GetId() ] = frame;
    }

    Frame::Ptr GetKeyframe(const size_t id) {
        unique_lock<mutex> lck(mutex_);
        return (keyframesDict_.count(id)) ? keyframesDict_[id] : nullptr;
    }

    KeyframeIdToPtr GetAllKeyframes() {
        unique_lock<mutex> lck(mutex_);
        return keyframesDict_;
    }

    void AddMappoint(const Mappoint::Ptr& map_point) {
        unique_lock<mutex> lck(mutex_);
        mappointsDict_[map_point->GetId()] = map_point;
    }

    Mappoint::Ptr GetMappoint(const size_t id) {
        unique_lock<mutex> lck(mutex_);
        return (mappointsDict_.count(id)) ? mappointsDict_[id] : nullptr;
    }

    MappointIdToPtr GetAllMappoints() {
        unique_lock<mutex> lck(mutex_);
        return mappointsDict_;
    }

    // Get the mpts observed by input keyframe, or its (active and non-active) covisible keyframes
    MappointIdToPtr GetMappointsNearKeyframe(const Frame::Ptr& keyframe);

    // Replace the old mappoint with new mappoint
    void ReplaceMappoint(size_t oldMptId, size_t newMptId);

    // If the old mpt was replaced by new mpt, get the respective new mpt
    Mappoint::Ptr GetPotentialReplacedMappoint(const size_t oldMptId);

private:
    mutex               mutex_;

    KeyframeIdToPtr     keyframesDict_;       // all key-frames
    MappointIdToPtr     mappointsDict_;       // all mappoints

    unordered_map<size_t, size_t> oldMptIdNewMptIdMap_;  // mpt mapping after replacement
};

} //namespace

#endif // MAP_H
