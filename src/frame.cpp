/*
 * Class frame maintains its RGB and depth image, and its pose in world reference frame
 *
 * When the instance is regarded as a keyframe, it will populate the observed mappoints and covisible keyframes, 
 * where the active keyframes share at least 15 covisible mappoints of this keyframe
 * 
 * After adding/removing an observed mappoints, the covisible keyframes will be updated automatcially
 */

#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/mapmanager.h"

namespace myslam
{

size_t Frame::factoryId_ = 0;

Frame::Ptr Frame::CreateFrame(
    const double timestamp, 
    const Camera::Ptr camera, 
    const Mat color, 
    const Mat depth)
{
    return Frame::Ptr( new Frame(
        ++factoryId_,
        move(timestamp),
        move(camera),
        color.clone(),
        depth.clone()) 
    );
}

Frame::Frame (  const size_t id, 
                const double timestamp, 
                const Camera::Ptr camera, 
                const Mat color, 
                const Mat depth )
: id_(move(id)), timestamp_(move(timestamp)), camera_(move(camera)), color_(move(color)), depth_(move(depth)), T_c_w_(SE3())
{

}

double Frame::GetDepth ( const KeyPoint& kp )
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return double(d)/camera_->GetDepthScale();
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return double(d)/camera_->GetDepthScale();
            }
        }
    }
    return -1.0;
}


bool Frame::IsInFrame ( const Vector3d& pt_world )
{
    Vector3d p_cam = camera_->World2Camera( pt_world, T_c_w_ );
    if ( p_cam(2, 0) < 0 ) {
        return false;
    } 
    Vector2d pixel = camera_->Camera2Pixel ( p_cam );
    return pixel(0,0)>0 && pixel(1,0)>0 
        && pixel(0,0)<color_.cols 
        && pixel(1,0)<color_.rows;
}

void Frame::AddObservedMappoint(const size_t mappointId, const Point2f pixelPos) {
    unique_lock<mutex> lck(observationMutex_);
    
    assert(!observedMappointIds_.count(mappointId));
    observedMappointIds_.insert(mappointId);

    auto mappoint = MapManager::GetInstance().GetMappoint(mappointId);
    assert(mappoint != nullptr);
    mappoint->AddObservedByKeyframe(this->id_, move(pixelPos));

    for (auto& idToPixel: mappoint->GetObservedByKeyframesMap()) {
        auto otherKFId = idToPixel.first;
        if (otherKFId == this->id_) {
            continue;
        }

        auto otherKF = MapManager::GetInstance().GetKeyframe(otherKFId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservedMappoint(mappointId));

        ++allCovisibleKeyframeIdToWeight_[otherKFId];
        if (allCovisibleKeyframeIdToWeight_[otherKFId] >= 15) {
            activeCovisibleKeyframes_.insert(otherKFId);
        }

        otherKF -> UpdateCovisibleKeyframeWeight(this->id_, allCovisibleKeyframeIdToWeight_[otherKFId]);
    }
    
}

void Frame::RemoveObservedMappoint(const size_t mappointId) {
    unique_lock<mutex> lck(observationMutex_);

    assert(observedMappointIds_.count(mappointId));
    observedMappointIds_.erase(mappointId);

    auto mappoint = MapManager::GetInstance().GetMappoint(mappointId);
    assert(mappoint != nullptr);
    mappoint->RemoveObservedByKeyframe(this->id_);

    for (auto& idToPixel: mappoint->GetObservedByKeyframesMap()) {
        auto otherKFId = idToPixel.first;
        if (otherKFId == this->id_) {
            continue;
        }

        auto otherKF = MapManager::GetInstance().GetKeyframe(otherKFId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservedMappoint(mappointId));

        --allCovisibleKeyframeIdToWeight_[otherKFId];
        if (allCovisibleKeyframeIdToWeight_[otherKFId] == 0) {
            allCovisibleKeyframeIdToWeight_.erase(otherKFId);
        } else if (activeCovisibleKeyframes_.count(otherKFId) && allCovisibleKeyframeIdToWeight_[otherKFId] < 15) {
            activeCovisibleKeyframes_.erase(otherKFId);
        }
        
        otherKF->UpdateCovisibleKeyframeWeight(this->id_, allCovisibleKeyframeIdToWeight_[otherKFId]);
    }

    // TODO: if all the observations has been removed, consider this keyframe as outlier?
}


void Frame::UpdateCovisibleKeyframeWeight(const size_t id, const int weight) {
    unique_lock<mutex> lck(observationMutex_);

    if (weight == 0) {
        allCovisibleKeyframeIdToWeight_.erase(id);
    } else if (weight >= 15) {
        allCovisibleKeyframeIdToWeight_[id] = weight;
        activeCovisibleKeyframes_.insert(id);
    } else {
        allCovisibleKeyframeIdToWeight_[id] = weight;
        if (activeCovisibleKeyframes_.count(id)) {
            activeCovisibleKeyframes_.erase(id);
        }
    }
}

}
