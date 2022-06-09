/*
 * Mappoint can only be created by factory function
 *
 * The observedBy keyframes field will be updated automatically by the keyframe instance
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
    const Mat          descriptor)
{
    // Mat is defaultly shadow copy
    return Mappoint::Ptr( 
        new Mappoint( 
            ++factoryId_, 
            move(position), 
            descriptor.clone())
    );
}


Mappoint::Mappoint ( 
    const size_t    id, 
    const Vector3d  position, 
    const Mat       descriptor)
: id_(move(id)), pos_(move(position)), descriptor_(move(descriptor)), norm_(Vector3d::Zero()),
    triangulated_(false), optimized_(false), outlier_(false) { }


void Mappoint::RemoveObservedByKeyframe(const size_t keyframeId) {
    unique_lock<mutex> lck(observationMutex_);
    assert(observedByKeyframeMap_.count(keyframeId));
    observedByKeyframeMap_.erase(keyframeId);

    // if all the observations has been removed
    if(observedByKeyframeMap_.size() == 0) {
        // cout << "Mark as outlier mappint: " << id_ << endl;
        outlier_ = true;
    }
}


} // namespace