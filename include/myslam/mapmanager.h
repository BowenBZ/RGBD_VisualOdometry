/*
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

    static MapManager& GetInstance() {
        static MapManager map_;
        return map_;
    }
    
    void InsertKeyframe( Frame::Ptr frame ) {
        unique_lock<mutex> lck(dataMutex_);
        keyframesDict_[ frame->GetId() ] = frame;
    }

    Frame::Ptr GetKeyframe( const size_t id ) {
        unique_lock<mutex> lck(dataMutex_);
        return (keyframesDict_.count(id)) ? keyframesDict_[id] : nullptr;
    }

    KeyframeIdToPtr GetAllKeyframes() {
        unique_lock<mutex> lck(dataMutex_);
        return keyframesDict_;
    }

    void InsertMappoint( Mappoint::Ptr map_point ) {
        unique_lock<mutex> lck(dataMutex_);
        mappointsDict_[map_point->GetId()] = map_point;
    }

    Mappoint::Ptr GetMappoint( const size_t id ) {
        unique_lock<mutex> lck(dataMutex_);
        return (mappointsDict_.count(id)) ? mappointsDict_[id] : nullptr;
    }

    MappointIdToPtr GetAllMappoints() {
        unique_lock<mutex> lck(dataMutex_);
        return mappointsDict_;
    }

    MappointIdToPtr GetMappointsAroundKeyframe( const Frame::Ptr& keyframe );


private:
    mutex               dataMutex_;

    MappointIdToPtr     mappointsDict_;       // all mappoints
    KeyframeIdToPtr     keyframesDict_;       // all key-frames
};

} //namespace

#endif // MAP_H
