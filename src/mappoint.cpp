#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

MapPoint::MapPoint ( 
    long unsigned int id, 
    const Vector3d& position, 
    const Vector3d& norm, 
    const Mat& descriptor,
    const unsigned long observedKeyFrameId,
    const cv::Point2f& pixelPos)
: id_(id), pos_(position), norm_(norm), triangulated_(false), visibleTimes_(1), matchedTimes_(1), descriptor_(descriptor), outlier_(false), optimized_(false)
{
    AddKeyframeObservation(observedKeyFrameId, pixelPos);
}

MapPoint::Ptr MapPoint::CreateMappoint ( 
    const Vector3d posWorld, 
    const Vector3d norm,
    const Mat descriptor,
    const unsigned long observedKeyFrameId,
    const cv::Point2f pixelPos)
{
    return MapPoint::Ptr( 
        new MapPoint( factoryId_++, posWorld, norm, descriptor, observedKeyFrameId, pixelPos)
    );
}

unsigned long MapPoint::factoryId_ = 0;


void MapPoint::removeKeyFrameObservation(const unsigned long keyFrameId) {
    unique_lock<mutex> lck(observationMutex_);
    for (auto iter = observedKeyFrameMap_.begin(); iter != observedKeyFrameMap_.end(); iter++) {
        if (iter->first == keyFrameId) {
            observedKeyFrameMap_.erase(iter);
            break;
        }
    }

    // if all the observations has been removed
    if(observedKeyFrameMap_.size() == 0) {
        // cout << "Mark as outlier mappint: " << id_ << endl;
        outlier_ = true;
    }
}


} // namespace