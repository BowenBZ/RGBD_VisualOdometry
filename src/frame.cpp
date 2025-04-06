#include "myslam/private/frame.hpp"

#include "myslam/private/util.hpp"
#include "myslam/private/mapmanager.hpp"
#include "myslam/private/superpoint_model.hpp"

#include <algorithm>

namespace myslam
{

size_t Frame::factoryId_ = 0;

Frame::Ptr Frame::CreateFrame(
    const struct FrameConfig& config,
    const double timestamp, 
    const Camera::Ptr& camera, 
    const cv::Mat& color, 
    const cv::Mat& depth)
{
    return Frame::Ptr( new Frame(
        config,
        factoryId_++,
        timestamp,
        camera,
        color,
        depth) 
    );
}

Frame::Frame (  const struct FrameConfig config,
                const size_t id, 
                const double timestamp, 
                const Camera::Ptr& camera, 
                const cv::Mat& color, 
                const cv::Mat& depth )
: id_(id), timestamp_(timestamp), camera_(camera), color_(color.clone()), depth_(depth.clone()), T_c_w_(SE3())
{
    config_ = config;
}

double Frame::GetDepth(const cv::KeyPoint& kp)
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if (d != 0) {
        return double(d) / camera_->GetDepthScale();
    }
    else {
        // Check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for (int i = 0; i < 4; i++)
        {
            d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
            if (d != 0) {
                return double(d) / camera_->GetDepthScale();
            }
        }
    }
    return -1.0;
}

#pragma mark - Feature extraction

void Frame::ExtractKeyPointsAndComputeDescriptors(const cv::Ptr<cv::Feature2D>& detector) {
    
    size_t rowPerSection = config_.imgRows / config_.rowSectionCnt;
    size_t colPerSection = config_.imgCols / config_.colSectionCnt;
    for(size_t rowSection = 0; rowSection < config_.rowSectionCnt; ++rowSection) {
        for (size_t colSection = 0; colSection < config_.colSectionCnt; ++colSection) {
            size_t rowStartIdx = rowPerSection * rowSection;
            size_t rowEndIdx = rowPerSection * (rowSection + 1);
            size_t colStartIdx = colPerSection * colSection;
            size_t colEndIdx = colPerSection * (colSection + 1);

            cv::Range rowRange(rowStartIdx, rowEndIdx);
            cv::Range colRange(colStartIdx, colEndIdx);

            std::vector<cv::KeyPoint> kpts;
            cv::Mat des;
            detector->detectAndCompute(color_(rowRange, colRange), cv::Mat(), kpts, des);

            for (size_t idx = 0; idx < std::min(kpts.size(), config_.maxFeaturesCnt / (config_.rowSectionCnt * config_.colSectionCnt)); ++idx) {
                auto& kpt = kpts[idx];
                kpt.pt.x += colStartIdx;
                kpt.pt.y += rowStartIdx;
                keypointInfo_.push_back({kpt, des.row(idx).clone(), std::nullopt});
                descriptors_.push_back(des.row(idx).clone());
            }
        }
    }
    
    ConstructKeypointGrids();

    // TODO: Extract more keypoints for keyframe
}

void Frame::ExtractKeypointsAndDescriptorsWithSuperPointModel(const SuperPointModel::Ptr model) {

    std::vector<CornerPoint> points;
    model->Process(color_, points, descriptors_);
    assert(points.size() == descriptors_.cols);

    // Each point's descriptor is in col of the output descriptors from model, we need to transpose it
    cv::transpose(descriptors_, descriptors_);

    for (size_t idx = 0; idx < points.size(); idx++) {
        cv::KeyPoint kpt(points[idx].x, points[idx].y, 1.f);
        keypointInfo_.push_back({kpt, descriptors_.row(idx).clone(), std::nullopt});
    }

    ConstructKeypointGrids();
}

void Frame::ConstructKeypointGrids() {
    for (size_t i = 0; i < keypointInfo_.size(); ++i) {
        auto& kptPos = keypointInfo_[i].keypoint.pt;
        size_t gridIdx = GetGridIdx(kptPos.x, kptPos.y);
        gridToKptIdx_[gridIdx].push_back(i);
    }
}

#pragma mark - Feature matching

bool Frame::SearchKeypointMatchCandidate(const Mappoint::Ptr& mpt, const bool doDirectionCheck, size_t& kptIdx, double& distance, bool& mayObserveMpt) {
    mayObserveMpt = false;

    Vector3d posInCam = T_c_w_ * mpt->GetPosition();
    if (posInCam[2] < 0) {
        return false;
    } 
    
    Vector2d pixelPos = camera_->Camera2Pixel(posInCam);
    if (pixelPos[0] < 0 || pixelPos[0] >= config_.imgCols ||
        pixelPos[1] < 0 || pixelPos[1] >= config_.imgRows) {
        return false;
    }

    if (doDirectionCheck) {
        Vector3d direction = mpt->GetPosition() - this->GetCamCenter();
        direction.normalize();
        double angle = acos( direction.transpose() * mpt->GetNormDirection() );
        if ( angle > M_PI / 6 ) {
            return false;
        }
    }

    // This frame may observe this mappoint, but not gurantee to have a matched keypoint
    // This is used by the fallback flann feature matching
    mayObserveMpt = true;
    
    const size_t mptGridIdx = GetGridIdx(pixelPos[0], pixelPos[1]);
    std::list<size_t> nearbyGrids;
    getNearbyGrids(mptGridIdx, nearbyGrids);
    std::vector<std::pair<size_t, double>> kptIdxToDistance;
    for (auto& gridIdx: nearbyGrids) {
        // This grid doesn't contain any keypoint
        if (!gridToKptIdx_.count(gridIdx)) {
            continue;
        }

        for (auto& kptIdx: gridToKptIdx_[gridIdx]) {
            double distance = ComputeDescriptorDistance(
                mpt->GetDescriptor(), 0, 
                keypointInfo_[kptIdx].descriptor, 0);

            kptIdxToDistance.push_back({kptIdx, distance});
        }
    }

    if (kptIdxToDistance.empty()) {
        return false;
    }

    sort(kptIdxToDistance.begin(), kptIdxToDistance.end(), 
        [](const std::pair<size_t, double>& kpt1, const std::pair<size_t, double>& kpt2) {
            return kpt1.second < kpt2.second;
        });

    const std::pair<size_t, double>& bestKptToDistance = kptIdxToDistance[0];
    if (bestKptToDistance.second > config_.descriptorDistanceThres) {
        return false;
    }

    if (kptIdxToDistance.size() >= 2) {
        const std::pair<size_t, double>& secondKptToDistance = kptIdxToDistance[1];
        if (bestKptToDistance.second / secondKptToDistance.second < config_.bestSecondaryDistanceRatio) {
            return false;
        }
    }
    
    kptIdx = bestKptToDistance.first;
    distance = bestKptToDistance.second;
    return true;
}

size_t Frame::GetGridIdx(double x, double y) {
    size_t colIdx = (size_t)floor(x / config_.gridSize);
    size_t rowIdx = (size_t)floor(y / config_.gridSize);
    return GetGridIdx(colIdx, rowIdx);
}

size_t Frame::GetGridIdx(size_t colIdx, size_t rowIdx) {
    return rowIdx * config_.gridColCnt + colIdx;
}

void Frame::getNearbyGrids(size_t gridIdx, std::list<size_t>& nearbyGrids) {
    nearbyGrids.clear();

    size_t rowIdx = gridIdx / config_.gridColCnt;
    size_t colIdx = gridIdx - rowIdx * config_.gridColCnt;

    for (int drow = -config_.searchGridRadius; drow <= config_.searchGridRadius; ++drow) {
        for (int dcol = -config_.searchGridRadius; dcol <= config_.searchGridRadius; ++dcol) {
            int row = (int)rowIdx + drow;
            int col = (int)colIdx + dcol;

            if (row >= 0 && row < config_.gridRowCnt &&
                col >= 0 && col < config_.gridColCnt) {
                    nearbyGrids.push_back(GetGridIdx((size_t)row, (size_t)col));
                }
        }
    }
}

#pragma mark - observing relationships

void Frame::AddObservingMappointCreatedFromOtherFrame(const size_t kptIdx, const Mappoint::Ptr& mpt) {
    auto& mptId = mpt->GetId();
    
    // Add the <kpt idx, mpt id> relationship
    assert(!keypointInfo_[kptIdx].optMatchedMptId.has_value());
    keypointInfo_[kptIdx].optMatchedMptId = mptId;

    assert(!observingMptIdToKptIdx_.count(mptId));
    observingMptIdToKptIdx_[mptId] = kptIdx;

    // Update mpt
    mpt->AddObservedByKeyframe(shared_from_this());
    mpt->UpdateDescriptor();

    // Update the only observation of the anchor keyframe id of the mappoint 
    assert(mpt->GetObservedByKeyframeIds().size() > 1);
    if (mpt->GetObservedByKeyframeIds().size() == 2) {
        const size_t anchorKfId = mpt->GetAnchoringKeyframeId();
        assert(anchorKfId != id_);
        auto anchorKf = MapManager::Instance().GetKeyframe(anchorKfId);
        anchorKf->RemoveOnlyThisObservedMpt(mptId);
    }

    // Update covisible graph
    for (auto& otherKfId: mpt->GetObservedByKeyframeIds()) {
        if (otherKfId == id_) {
            continue;
        }

        auto otherKF = MapManager::Instance().GetKeyframe(otherKfId);
        assert(otherKF->IsObservingMappoint(mptId));

        auto& covisibleWeight = allCovisibleKfIdToWeight_[otherKfId];
        ++covisibleWeight;
        if (covisibleWeight >= config_.activeCovisibleWeight) {
            activeCovisibleKfIds_.insert(otherKfId);
        }

        otherKF->UpdateCovisibleKeyframeWeight(id_, covisibleWeight);
    }
}

void Frame::AddObservingMappointCreatedFromThisFrame(const size_t kptIdx, const Mappoint::Ptr& mpt) {
    auto& mptId = mpt->GetId();

    // Add the <kpt idx, mpt id> relationship
    assert(!keypointInfo_[kptIdx].optMatchedMptId.has_value());
    keypointInfo_[kptIdx].optMatchedMptId = mptId;

    assert(!observingMptIdToKptIdx_.count(mptId));
    observingMptIdToKptIdx_[mptId] = kptIdx;

    // Add this mpt id to new observed set, since this mappoint is created from this keyframe
    AddOnlyThisObservedMpt(mptId);

    // Update mpt
    mpt->AddAnchoringKeyframeId(id_);
    mpt->AddObservedByKeyframe(shared_from_this());

    // No need to update covisible graph since this mappoint is only observed by this frame.
}

void Frame::RemoveObservingMappointCreatedFromOtherFrame(const size_t mptId) {
    // Remove the <kpt idx, mpt id> relationship
    assert(observingMptIdToKptIdx_.count(mptId));
    size_t kptIdx = observingMptIdToKptIdx_[mptId];

    assert(keypointInfo_[kptIdx].optMatchedMptId.has_value());
    keypointInfo_[kptIdx].optMatchedMptId = std::nullopt;

    observingMptIdToKptIdx_.erase(mptId);

    // Remove the observedBy relationship from the mappoint
    // Note the observation from anchor keyframe cannot be removed
    auto mpt = MapManager::Instance().GetMappoint(mptId);
    
    // This mappoint may already get removed
    if (mpt == nullptr) {
        return;
    }

    assert(mpt != nullptr);
    const size_t anchorKfId = mpt->GetAnchoringKeyframeId();
    assert(anchorKfId != id_);
    mpt->RemoveObservedByKeyframe(this->id_);

    if (mpt->GetObservedByKeyframeIds().size() == 1) {
        auto anchorKf = MapManager::Instance().GetKeyframe(anchorKfId);
        anchorKf->AddOnlyThisObservedMpt(mptId);
    }

    // Update covisible graph
    for (auto& otherKFId: mpt->GetObservedByKeyframeIds()) {
        if (otherKFId == this->id_) {
            continue;
        }

        auto otherKF = MapManager::Instance().GetKeyframe(otherKFId);
        assert(otherKF->IsObservingMappoint(mptId));

        auto& covisibleWeight = allCovisibleKfIdToWeight_[otherKFId];
        --covisibleWeight;
        if (covisibleWeight == 0) {
            allCovisibleKfIdToWeight_.erase(otherKFId);
        } else if (covisibleWeight < config_.activeCovisibleWeight) {
            activeCovisibleKfIds_.erase(otherKFId);
        }
        
        otherKF->UpdateCovisibleKeyframeWeight(this->id_, covisibleWeight);
    }

    // TODO: if all the observations has been removed, consider this keyframe as outlier?
}

void Frame::RemoveObservingMappointCreatedFromThisFrame(const size_t mptId) {
    // Remove the <kpt idx, mpt id> relationship
    assert(observingMptIdToKptIdx_.count(mptId));
    size_t kptIdx = observingMptIdToKptIdx_[mptId];

    assert(keypointInfo_[kptIdx].optMatchedMptId.has_value());
    keypointInfo_[kptIdx].optMatchedMptId = std::nullopt;

    observingMptIdToKptIdx_.erase(mptId);

    // Remove the only observation
    assert(onlyThisObservedMptId_.count(mptId));
    onlyThisObservedMptId_.erase(mptId);

    // No need to operate on the mpt itself since it will be removed
}

#pragma mark - covisible keyframes

void Frame::UpdateCovisibleKeyframeWeight(const size_t otherKfId, const size_t weight) {
    if (weight == 0) {
        allCovisibleKfIdToWeight_.erase(otherKfId);
        activeCovisibleKfIds_.erase(otherKfId);
    } else if (weight >= config_.activeCovisibleWeight) {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        activeCovisibleKfIds_.insert(otherKfId);
    } else {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        activeCovisibleKfIds_.erase(otherKfId);
    }
}

}
