#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

size_t MapPoint::factoryId_ = 0;

MapPoint::Ptr MapPoint::CreateMappoint ( 
    const Vector3d&     position, 
    const Vector3d&     norm,
    const Mat           descriptor,
    const size_t        observedByKeyframeId,
    const cv::Point2f&  pixelPos)
{
    // Mat is defaultly shadow copy
    return MapPoint::Ptr( 
        new MapPoint( factoryId_++, position, norm, descriptor, observedByKeyframeId, pixelPos)
    );
}


MapPoint::MapPoint ( 
    const size_t        id, 
    const Vector3d&     position, 
    const Vector3d&     norm, 
    const Mat           descriptor,
    const size_t        observedByKeyframeId,
    const cv::Point2f&  pixelPos)
: id_(id), pos_(position), norm_(norm), descriptor_(descriptor.clone()), 
    triangulated_(false), optimized_(false), outlier_(false), visibleTimes_(1), matchedTimes_(1)
{
    AddKeyframeObservation(observedByKeyframeId, pixelPos);
}


void MapPoint::RemoveObservedByKeyframe(const unsigned long keyFrameId) {
    unique_lock<mutex> lck(observationMutex_);
    for (auto iter = observedByKeyframeMap_.begin(); iter != observedByKeyframeMap_.end(); iter++) {
        if (iter->first == keyFrameId) {
            observedByKeyframeMap_.erase(iter);
            break;
        }
    }

    // if all the observations has been removed
    if(observedByKeyframeMap_.size() == 0) {
        // cout << "Mark as outlier mappint: " << id_ << endl;
        outlier_ = true;
    }
}


} // namespace