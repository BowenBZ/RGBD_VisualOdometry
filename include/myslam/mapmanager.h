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
        keyframesDict_[ frame->GetId() ] = std::move(frame);
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
        mappointsDict_[map_point->GetId()] = std::move(map_point);
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


    // // cull the hardly seen and no visible points of current frame from active mappoints
    // void cullNonActiveMapPoints( const Frame::Ptr& currFrame );

    // void updateMappointEraseRatio() {
    //     mapPointEraseRatio_ = (activeMapPoints_.size() > 1000 ) ? 
    //                                 mapPointEraseRatio_ + 0.05 :
    //                                 0.1;
    // }


private:
    MapManager() {   
        mapPointEraseRatio_ = Config::get<double> ( "map_point_erase_ratio" );
    }

    mutex               dataMutex_;

    MappointIdToPtr     mappointsDict_;       // all mappoints
    KeyframeIdToPtr     keyframesDict_;       // all key-frames

    float               mapPointEraseRatio_;
};

} //namespace

#endif // MAP_H
