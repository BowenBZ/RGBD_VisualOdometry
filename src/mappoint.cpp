/*
 * Mappoint can only be created by factory function
 *
 * The observedBy keyframes field will be updated automatically by the keyframe instance
 *  
 * Mappoint uses different mutex when modifying the position or the observedBy relationships
 */

#include "myslam/common_include.h"
#include "myslam/util.h"
#include "myslam/mappoint.h"
#include "myslam/frame.h"
#include "myslam/mapmanager.h"

namespace myslam
{

size_t Mappoint::factoryId_ = 0;

Mappoint::Ptr Mappoint::CreateMappoint(const Vector3d& pos)
{
    // Vector3d is deep copy, while Mat is shadow copy
    return Mappoint::Ptr( 
        new Mappoint(++factoryId_, pos)
    );
}


Mappoint::Mappoint(const size_t id, const Vector3d& pos)
: id_(id), pos_(pos), norm_(Vector3d::Zero()),
  triangulated_(false), optimized_(false), outlier_(false) { }


void Mappoint::AddObservedByKeyframe(const Frame::Ptr& kf, const size_t kptIdx) {
    unique_lock<mutex> lock(observationMutex_);
    auto kfId = kf->GetId();
    assert(!observedByKfIdToKptIdx_.count(kfId));

    observedByKfIdToKptIdx_[kfId] = kptIdx;

    // Calculate mpt descriptor
    CalculateMappointDescriptor();

    // Calculate mpt average direction
    auto direction = (pos_ - kf->GetCamCenter()).normalized();
    norm_ = (norm_ + direction).normalized();
}

void Mappoint::CalculateMappointDescriptor() {
    // Get all matched keypoint descriptors for this mappoint
    vector<Mat> descriptors;
    size_t desCnt = descriptors.size();
    descriptors.reserve(desCnt);
    for(auto& [kfId, kptIdx]: observedByKfIdToKptIdx_) {
        auto keyframe = MapManager::GetInstance().GetKeyframe(kfId);
        descriptors.push_back(keyframe->GetDescriptor(kptIdx));
    }

    // Calculate the distance between descriptors
    vector<vector<double>> descriptorDistances(desCnt, vector<double>(desCnt, 0));
    for(size_t i = 0; i < desCnt; ++i) {
        for(size_t j = i + 1; j < desCnt; ++j) {
            double distance = ComputeDescriptorDistance(
                descriptors[i], 0,
                descriptors[j], 0
            );
            descriptorDistances[i][j] = distance;
            descriptorDistances[j][i] = distance;
        }
    }

    // Calculate the medium distance from each descriptor to the others, and select the minumum
    double minMedium = numeric_limits<double>::max();
    double desIdx = 0;
    for(size_t i = 0; i < desCnt; ++i) {
        auto& distances = descriptorDistances[i];
        sort(distances.begin(), distances.end());
        double medium = (desCnt % 2 == 0) ?
            (distances[desCnt / 2 - 1] + distances[desCnt / 2]) / 2.0:
            distances[desCnt / 2];
        
        if (medium < minMedium) {
            minMedium = medium;
            desIdx = i;
        }
    }

    // Set the mappoint descriptor
    descriptor_ = descriptors[desIdx];
}

void Mappoint::RemoveObservedByKeyframe(const size_t keyframeId) {
    unique_lock<mutex> lck(observationMutex_);
    assert(observedByKfIdToKptIdx_.count(keyframeId));
    observedByKfIdToKptIdx_.erase(keyframeId);

    // if all the observations has been removed
    if(observedByKfIdToKptIdx_.size() == 0) {
        // cout << "Mark as outlier mappint: " << id_ << endl;
        outlier_ = true;
    } else {
        CalculateMappointDescriptor();
    }
}


} // namespace