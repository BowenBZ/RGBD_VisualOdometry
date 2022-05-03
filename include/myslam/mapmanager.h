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
    typedef unordered_map<unsigned long, MapPoint::Ptr > MappointDict;
    typedef unordered_map<unsigned long, Frame::Ptr > KeyframeDict;

    static MapManager& GetInstance() {
        static MapManager map_;
        return map_;
    }
    
    void InsertKeyframe( Frame::Ptr frame ) {
        unique_lock<mutex> lck(dataMutex_);
        keyframesDict_[ frame->getId() ] = std::move(frame);
    }

    void InsertMappoint( MapPoint::Ptr map_point ) {
        unique_lock<mutex> lck(dataMutex_);
        mappointsDict_[map_point->getId()] = std::move(map_point);
    }

    KeyframeDict GetAllKeyframes() {
        unique_lock<mutex> lck(dataMutex_);
        return keyframesDict_;
    }

    MappointDict GetAllMappoints() {
        unique_lock<mutex> lck(dataMutex_);
        return mappointsDict_;
    }

    Frame::Ptr GetKeyframe(unsigned long id) {
        unique_lock<mutex> lck(dataMutex_);
        return (keyframesDict_.count(id)) ? keyframesDict_[id] : nullptr;
    }

    MappointDict GetMappointsAroundKeyframe( const Frame::Ptr& keyframe );

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
    ~MapManager() {
    }

    mutex           dataMutex_;

    MappointDict    mappointsDict_;       // all mappoints
    KeyframeDict    keyframesDict_;       // all key-frames

    float mapPointEraseRatio_;
};

} //namespace

#endif // MAP_H
