#include "myslam/private/mappoint.hpp"

#include "myslam/private/frame.hpp"
#include "myslam/private/mapmanager.hpp"
#include "myslam/private/util.hpp"

#include <myslam/common_include.hpp>

namespace myslam
{

size_t Mappoint::factoryId_ = 0;

Mappoint::Ptr Mappoint::CreateMappoint(const Vector3d& pos, const cv::Mat& descriptor, const bool superpointEnabled)
{
    // Vector3d is deep copy, while cv::Mat is shadow copy
    return Mappoint::Ptr( 
        new Mappoint(++factoryId_, pos, descriptor, superpointEnabled)
    );
}


Mappoint::Mappoint(const size_t id, const Vector3d& pos, const cv::Mat& descriptor, const bool superpointEnabled)
: id_(id), pos_(pos), descriptor_(descriptor.clone()), norm_(Vector3d::Zero()),
  superpointEnabled_(superpointEnabled) { }


void Mappoint::AddObservedByKeyframe(const std::shared_ptr<Frame>& kf) {
    const auto& kfId = kf->GetId();
    assert(!observedByKfId_.count(kfId));

    observedByKfId_.insert(kfId);

    // Update the mpt average viewing direction
    auto direction = (pos_ - kf->GetCamCenter()).normalized();
    norm_ = (norm_ + direction).normalized();

    // TODO: need to update descriptor at this step?
}

void Mappoint::RemoveObservedByKeyframe(const size_t keyframeId) {
    assert(observedByKfId_.count(keyframeId));
    observedByKfId_.erase(keyframeId);

    // TODO: need to update descriptor at this step? 
}

void Mappoint::UpdateNormViewDirection() {
    Vector3d direction(0, 0, 0);
    for(auto& kfId: observedByKfId_) {
        auto kf = MapManager::Instance().GetKeyframe(kfId);
        direction += (pos_ - kf->GetCamCenter()).normalized();
    }
    norm_ = direction.normalized();
}

void Mappoint::UpdateDescriptor() {
    // When the observed by keyframe is less than 2, no need to calculate
    if (observedByKfId_.size() <= 2) {
        return;
    }

    // Get all matched keypoint descriptors for this mappoint
    std::vector<cv::Mat> descriptors;
    size_t desCnt = descriptors.size();
    descriptors.reserve(desCnt);
    for(auto& kfId: observedByKfId_) {
        auto kf = MapManager::Instance().GetKeyframe(kfId);
        const auto& optkptIdx = kf->GetMatchedKeypointIdxForMappoint(id_);
        assert(optkptIdx.has_value());
        descriptors.push_back(kf->GetDescriptor(optkptIdx.value()));
    }

    // Calculate the distance between descriptors
    std::vector<std::vector<double>> descriptorDistances(desCnt, std::vector<double>(desCnt, 0));
    for(size_t i = 0; i < desCnt; ++i) {
        for(size_t j = i + 1; j < desCnt; ++j) {
            double distance = superpointEnabled_ ? 
                ComputeSuperpointDescriptorL2Distance(descriptors[i], descriptors[j]): 
                ComputeDescriptorHammingDistance(descriptors[i], 0, descriptors[j], 0);
            descriptorDistances[i][j] = distance;
            descriptorDistances[j][i] = distance;
        }
    }

    // Calculate the medium distance from each descriptor to the others, and select the minumum
    double minMedium = std::numeric_limits<double>::max();
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

} // namespace