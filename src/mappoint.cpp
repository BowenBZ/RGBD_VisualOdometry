/*
 * Mappoint can only be created by factory function
 *
 * Mappoint maintain the relationship with keyframes who observe this mappoint
 *  
 * Mappoint uses different mutex when modifying the position or the observedBy relationships
 */

#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

size_t Mappoint::factoryId_ = 0;

Mappoint::Ptr Mappoint::CreateMappoint ( 
    const Vector3d     position, 
    const Vector3d     norm,
    const Mat          descriptor,
    const size_t       observedByKeyframeId,
    const Point2f      pixelPos)
{
    // Mat is defaultly shadow copy
    return Mappoint::Ptr( 
        new Mappoint( 
            ++factoryId_, 
            move(position), 
            move(norm), 
            descriptor.clone(), 
            move(observedByKeyframeId), 
            move(pixelPos))
    );
}


Mappoint::Mappoint ( 
    const size_t    id, 
    const Vector3d  position, 
    const Vector3d  norm, 
    const Mat       descriptor,
    const size_t    observedByKeyframeId,
    const Point2f   pixelPos)
: id_(move(id)), pos_(move(position)), norm_(move(norm)), descriptor_(move(descriptor)), 
    triangulated_(false), optimized_(false), outlier_(false)
{
    AddKeyframeObservation(move(observedByKeyframeId), move(pixelPos));
}


void Mappoint::RemoveObservedByKeyframe(const size_t keyframeId) {
    unique_lock<mutex> lck(observationMutex_);
    if (observedByKeyframeMap_.count(keyframeId)) {
        observedByKeyframeMap_.erase(keyframeId);
    }

    // if all the observations has been removed
    if(observedByKeyframeMap_.size() == 0) {
        // cout << "Mark as outlier mappint: " << id_ << endl;
        outlier_ = true;
    }
}


} // namespace