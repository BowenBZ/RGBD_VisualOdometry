/*
 * Class frame maintains its RGB and depth image, and its pose in world reference frame
 *
 * When the instance is regarded as a keyframe, it will populate the observed mappoints and covisible keyframes, 
 * where the keyframes share at least 15 covisible mappoints of this keyframe
 * 
 * After adding all observed mappoints, the covisible keyframes need to be called manually to be computed
 * When removing an observed mappoints, the covisible keyframes will be updated automatically
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

void Frame::RemoveObservedMappoint(const size_t id) {
    unique_lock<mutex> lck(observationMutex_);

    if (!observedMappointIds_.count(id)) {
        return;
    }

    observedMappointIds_.erase(id);

    auto mappoint = MapManager::GetInstance().GetMappoint(id);
    if (mappoint == nullptr) {
        return;
    }

    for (auto& idToPixel: mappoint->GetObservedByKeyframesMap()) {
        auto otherKFId = idToPixel.first;
        auto otherKF = MapManager::GetInstance().GetKeyframe(otherKFId);

        if ( otherKF == nullptr || otherKF->GetId() == this->id_ ) {
            continue;
        }

        // if the other keyframe has already removed the observation of this mappoint
        if ( !otherKF->IsObservedMappoint(id) ) {
            continue;
        }

        // update both the covisible keyframes for this keyframe and the other keyframe
        this   ->DecreaseCovisibleKeyFrameWeightByOneWithoutMutex(otherKFId);
        otherKF->DecreaseCovisibleKeyframeWeightByOne(this->id_);
    }

    // if all the observations has been removed, consider this keyframe as outlier?
}

void Frame::DecreaseCovisibleKeyframeWeightByOne(const size_t id) {
    unique_lock<mutex> lck(observationMutex_);

    DecreaseCovisibleKeyFrameWeightByOneWithoutMutex(id);
}

void Frame::DecreaseCovisibleKeyFrameWeightByOneWithoutMutex(const size_t id) {
    if (covisibleKeyframeIdToWeight_.count(id)) {
        covisibleKeyframeIdToWeight_[id]--;
        if (covisibleKeyframeIdToWeight_[id] < 15 
            && covisibleKeyframeIdToWeight_.size() >= 2) {
                covisibleKeyframeIdToWeight_.erase(id);
            }
    }
}

void Frame::ComputeCovisibleKeyframes() {
    unique_lock<mutex> lck(observationMutex_);

    // Calcualte the co-visibliblity keyframes' weight
    CovisibleKeyframeIdToWeight covisibleKeyframeIdToWeightCandidates;
    for(auto& mappointId : observedMappointIds_) {
        auto mappoint = MapManager::GetInstance().GetMappoint(mappointId);
        if (mappoint == nullptr || mappoint->outlier_) {
            continue;
        }

        for(auto& idToPixelPos : mappoint->GetObservedByKeyframesMap()) {
            auto keyframe = MapManager::GetInstance().GetKeyframe(idToPixelPos.first);
            if ( keyframe == nullptr || keyframe->GetId() == id_) {
                continue;
            }

            ++covisibleKeyframeIdToWeightCandidates[keyframe->GetId()];
        }
    }
       
    // Filter the connections whose weight larger than 15
    covisibleKeyframeIdToWeight_.clear();
    size_t idWithMaxWeight;
    int maxWeight = 0;

    for(auto& idToWeight : covisibleKeyframeIdToWeightCandidates) {
        if(idToWeight.second >= 15) {
            covisibleKeyframeIdToWeight_[idToWeight.first] = idToWeight.second;
            MapManager::GetInstance().GetKeyframe(idToWeight.first)->AddCovisibleKeyframe(this->id_, idToWeight.second);
        }

        if (idToWeight.second > maxWeight) {
            idWithMaxWeight = idToWeight.first;
            maxWeight = idToWeight.second;
        }
    }

    // In case there is no weight larger than 15
    if(covisibleKeyframeIdToWeight_.empty() && maxWeight != 0) {
        covisibleKeyframeIdToWeight_[idWithMaxWeight] = maxWeight;
        MapManager::GetInstance().GetKeyframe(idWithMaxWeight)->AddCovisibleKeyframe(this->id_, maxWeight);
    }

    // cout << "Current Keyframe Id: " << this->id_ << endl;
    // cout << "Connected Keyframe counts: " << covisibleKeyframeIdToWeight_.size() << endl;
    // for(auto& pair: covisibleKeyframeIdToWeight_) {
    //     cout << "Id: " << pair.first << " Weight: " << pair.second << endl;
    // }    
}


}
