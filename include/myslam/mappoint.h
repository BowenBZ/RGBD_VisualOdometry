/*
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{
class Frame;

class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef shared_ptr<MapPoint> Ptr;
    typedef list<pair<size_t, cv::Point2f>> ObservedByKFIdtoPixelPos;

    Mat         descriptor_;            // Descriptor for matching 
    Vector3d    norm_;                  // Normal of viewing direction 
    int         visibleTimes_;          // times should in the view of current frame, but maybe cannot be matched 
    int         matchedTimes_;          // times of being an inliner in frontend P3P result

    bool        triangulated_;          // whether have been triangulated in frontend
    bool        optimized_;             // whether is optimized by backend

    // Usages 
    // 1. whether match with new mappoints in front end
    // 2. whether will be triangulated
    // 3. whether add new observation keyframe 
    // 4. whether update in co-visibility graph
    // 5. whether add into backend
    bool        outlier_;               // whether this is an outlider

    // factory function to create mappoint
    static MapPoint::Ptr CreateMappoint( 
        const Vector3d&     position, 
        const Vector3d&     norm,
        const Mat           descriptor,
        size_t              observedByKeyframeId,
        const cv::Point2f&  pixelPos);

    Vector3d GetPosition() {
        unique_lock<mutex> lock(posMutex_);
        return pos_;
    }

    void SetPosition(const Vector3d& pos) {
        unique_lock<mutex> lock(posMutex_);
        pos_ = pos;
    }

    size_t GetId() { 
        return id_; 
    }

    void AddKeyframeObservation(size_t keyFrameId, const cv::Point2f& pixel_pos) {
        unique_lock<mutex> lock(observationMutex_);
        observedByKeyframeMap_.push_back(make_pair(keyFrameId, pixel_pos));
    }

    ObservedByKFIdtoPixelPos GetObservedByKeyframesMap() {
        unique_lock<mutex> lock(observationMutex_);
        return observedByKeyframeMap_;
    }

    void RemoveObservedByKeyframe(size_t keyFrameId);

private:
    static size_t               factoryId_;
    size_t                      id_;

    mutex                       posMutex_;
    Vector3d                    pos_;       // Position in world reference frame

    mutex                       observationMutex_;
    ObservedByKFIdtoPixelPos    observedByKeyframeMap_;

    // mappoint can only be created by factory
    MapPoint( 
        size_t              id, 
        const Vector3d&     position, 
        const Vector3d&     norm, 
        const Mat           descriptor,
        size_t              observedByKeyframeId,
        const cv::Point2f&  pixelPos);
};

} // namespace

#endif // MAPPOINT_H
