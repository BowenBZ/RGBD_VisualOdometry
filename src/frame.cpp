#include "myslam/private/frame.hpp"

#include "myslam/config.hpp"
#include "myslam/private/util.hpp"
#include "myslam/private/mapmanager.hpp"
#include "myslam/private/superpoint_model.hpp"

#include <algorithm>

namespace myslam
{

#pragma mark - Frame Config

FrameConfig::FrameConfig() {
    imgCols = Config::get<int>("frame.width");
    imgRows = Config::get<int>("frame.height");
    activeCovisibleWeight = Config::get<int>("frame.active_covisible_keyframe_weight");

    orbConfig.numOfFeatures = Config::get<int>("frame.orb.number_of_features");
    orbConfig.scaleFactor = Config::get<float>("frame.orb.scale_factor");
    orbConfig.levelPyramid = Config::get<int>("frame.orb.level_pyramid");
    orbConfig.rowSectionCnt = Config::get<int>("frame.orb.row_section_cnt");
    orbConfig.colSectionCnt = Config::get<int>("frame.orb.col_section_cnt");

    searchConfig.gridSize = Config::get<int>("frame.active_search.grid_size");
    searchConfig.gridColCnt = ceil((double)imgCols / searchConfig.gridSize);
    searchConfig.gridRowCnt = ceil((double)imgRows / searchConfig.gridSize);
    searchConfig.searchGridRadius = Config::get<int>("frame.active_search.search_grid_radius");
    searchConfig.descriptorMatchDistanceThreshORB = Config::get<double>("frame.active_search.max_matched_descriptor_distance_orb");
    searchConfig.secondaryBestMatchDistanceRatioORB = Config::get<double>("frame.active_search.min_best_secondary_distance_ratio_orb");
}

#pragma mark - keypoint info

struct GridInfo {
    size_t id;
    size_t rowIdx;
    size_t colIdx;
};

#pragma mark - class Frame

size_t Frame::factoryId_ = 0;

Frame::Ptr Frame::CreateFrame(
    const std::shared_ptr<struct FrameConfig> config,
    const double timestamp, 
    const Camera::Ptr& camera, 
    const cv::Mat& color, 
    const cv::Mat& depth)
{
    return Frame::Ptr(new Frame(
        config,
        factoryId_++,
        timestamp,
        camera,
        color,
        depth) 
    );
}

Frame::Frame (  const std::shared_ptr<struct FrameConfig> config,
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

void Frame::ExtractKeyPointsAndComputeDescriptorsORB(const cv::Ptr<cv::Feature2D>& detector) {
    
    size_t rowPerSection = config_->imgRows / config_->orbConfig.rowSectionCnt;
    size_t colPerSection = config_->imgCols / config_->orbConfig.colSectionCnt;
    for(size_t rowSection = 0; rowSection < config_->orbConfig.rowSectionCnt; ++rowSection) {
        for (size_t colSection = 0; colSection < config_->orbConfig.colSectionCnt; ++colSection) {
            size_t rowStartIdx = rowPerSection * rowSection;
            size_t rowEndIdx = rowPerSection * (rowSection + 1);
            size_t colStartIdx = colPerSection * colSection;
            size_t colEndIdx = colPerSection * (colSection + 1);

            cv::Range rowRange(rowStartIdx, rowEndIdx);
            cv::Range colRange(colStartIdx, colEndIdx);

            std::vector<cv::KeyPoint> kpts;
            cv::Mat des;
            detector->detectAndCompute(color_(rowRange, colRange), cv::Mat(), kpts, des);

            for (size_t idx = 0; idx < std::min(kpts.size(), config_->orbConfig.numOfFeatures / (config_->orbConfig.rowSectionCnt * config_->orbConfig.colSectionCnt)); ++idx) {
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

#pragma mark - Feature matching

std::optional<KptMatchResult> Frame::SearchORBKeypointMatchCandidate(const Mappoint::Ptr& mpt, const bool doDirectionCheck) {
    Vector3d posInCam = T_c_w_ * mpt->GetPosition();
    if (posInCam[2] < 0) {
        return std::nullopt;
    } 
    
    Vector2d pixelPos = camera_->Camera2Pixel(posInCam);
    if (pixelPos[0] < 0 || pixelPos[0] >= config_->imgCols ||
        pixelPos[1] < 0 || pixelPos[1] >= config_->imgRows) {
        return std::nullopt;
    }

    if (doDirectionCheck) {
        Vector3d direction = mpt->GetPosition() - this->GetCamCenter();
        direction.normalize();
        double angle = acos( direction.transpose() * mpt->GetNormDirection() );
        if ( angle > M_PI / 6 ) {
            return std::nullopt;
        }
    }
    
    GridInfo gridInfo;
    KeypointPosToGridInfo(pixelPos[0], pixelPos[1], gridInfo);
    std::list<GridInfo> nearbyGrids;
    GetNearbyGrids(gridInfo, nearbyGrids);
    std::vector<KptMatchResult> matchResults;
    for (auto& nearbyGrid: nearbyGrids) {
        // This grid doesn't contain any keypoint
        if (!gridToKptIdx_.count(nearbyGrid.id)) {
            continue;
        }

        for (auto& kptIdx: gridToKptIdx_[nearbyGrid.id]) {
            const float distance = ComputeDescriptorHammingDistance(
                mpt->GetDescriptor(), 0, 
                keypointInfo_[kptIdx].descriptor, 0);

            matchResults.push_back({kptIdx, distance});
        }
    }

    if (matchResults.empty()) {
        return std::nullopt;
    }

    sort(matchResults.begin(), matchResults.end(), 
        [](const KptMatchResult& res1, const KptMatchResult& res2) {
            return res1.distance < res2.distance;
        });

    const auto& bestMatch = matchResults[0];
    if (bestMatch.distance > config_->searchConfig.descriptorMatchDistanceThreshORB) {
        return std::nullopt;
    }

    if (matchResults.size() >= 2) {
        const auto& secondBestMatch = matchResults[1];
        if (bestMatch.distance / secondBestMatch.distance < config_->searchConfig.secondaryBestMatchDistanceRatioORB) {
            return std::nullopt;
        }
    }
    
    return bestMatch;
}

std::optional<KptMatchResult> Frame::SearchSuperpointKeypointMatchCandidate(const Mappoint::Ptr& mpt, const float distanceMatchThresh, const std::optional<float> distanceRatioThresh) {
    Vector3d posInCam = T_c_w_ * mpt->GetPosition();
    if (posInCam[2] < 0) {
        return std::nullopt;
    } 
    
    Vector2d pixelPos = camera_->Camera2Pixel(posInCam);
    if (pixelPos[0] < 0 || pixelPos[0] >= config_->imgCols ||
        pixelPos[1] < 0 || pixelPos[1] >= config_->imgRows) {
        return std::nullopt;
    }

    /*
        Vector3d direction = mpt->GetPosition() - this->GetCamCenter();
        direction.normalize();
        double angle = acos( direction.transpose() * mpt->GetNormDirection() );
        if ( angle > M_PI / 6 ) {
            return false;
        }
    */

    GridInfo gridInfo;
    KeypointPosToGridInfo(pixelPos[0], pixelPos[1], gridInfo);
    std::list<GridInfo> nearbyGrids;
    GetNearbyGrids(gridInfo, nearbyGrids);

    std::vector<KptMatchResult> matchResults;
    for (auto& nearbyGridInfo: nearbyGrids) {
        // This grid doesn't contain any keypoint
        if (!gridToKptIdx_.count(nearbyGridInfo.id)) {
            continue;
        }

        for (auto& kptIdx: gridToKptIdx_[nearbyGridInfo.id]) {
            float distance = ComputeSuperpointDescriptorL2Distance(
                mpt->GetDescriptor(), keypointInfo_[kptIdx].descriptor);

            matchResults.push_back({kptIdx, distance});
        }
    }

    if (matchResults.empty()) {
        return std::nullopt;
    }

    sort(matchResults.begin(), matchResults.end(), 
        [](const KptMatchResult& res1, const KptMatchResult& res2) {
            return res1.distance < res2.distance;
        });

    const auto& bestMatch = matchResults[0];
    if (bestMatch.distance > distanceMatchThresh) {
        return std::nullopt;
    }

    if (distanceRatioThresh.has_value() && matchResults.size() >= 2) {
        const auto& secondBestMatch = matchResults[1];
        if (bestMatch.distance / secondBestMatch.distance < distanceRatioThresh.value()) {
            return std::nullopt;
        }
    }

    return bestMatch;
}

#pragma mark - Grid indexing

void Frame::ConstructKeypointGrids() {
    for (size_t i = 0; i < keypointInfo_.size(); ++i) {
        auto& kptPos = keypointInfo_[i].keypoint.pt;
        GridInfo gridInfo;
        KeypointPosToGridInfo(kptPos.x, kptPos.y, gridInfo);
        gridToKptIdx_[gridInfo.id].push_back(i);
    }
}

void Frame::KeypointPosToGridInfo(const double x, const double y, GridInfo& gridInfo) {
    gridInfo.colIdx = (size_t)floor(x / config_->searchConfig.gridSize);
    gridInfo.rowIdx = (size_t)floor(y / config_->searchConfig.gridSize);
    PopulateGridId(gridInfo);
}

void Frame::PopulateGridId(GridInfo& gridInfo) {
    gridInfo.id = gridInfo.rowIdx * config_->searchConfig.gridColCnt + gridInfo.colIdx;
}

void Frame::GetNearbyGrids(const GridInfo& gridInfo, std::list<GridInfo>& nearbyGrids) {
    nearbyGrids.clear();

    for (int drow = -config_->searchConfig.searchGridRadius; drow <= config_->searchConfig.searchGridRadius; ++drow) {
        for (int dcol = -config_->searchConfig.searchGridRadius; dcol <= config_->searchConfig.searchGridRadius; ++dcol) {
            int row = (int)(gridInfo.rowIdx) + drow;
            int col = (int)(gridInfo.colIdx) + dcol;

            if (row >= 0 && row < config_->searchConfig.gridRowCnt &&
                col >= 0 && col < config_->searchConfig.gridColCnt) {
                    GridInfo gridInfo;
                    gridInfo.rowIdx = row;
                    gridInfo.colIdx = col;
                    PopulateGridId(gridInfo);
                    nearbyGrids.push_back(gridInfo);
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
        if (covisibleWeight >= config_->activeCovisibleWeight) {
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
        } else if (covisibleWeight < config_->activeCovisibleWeight) {
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

    // Remove the only observation. Note this mpt may be observed by other frames
    RemoveOnlyThisObservedMpt(mptId);

    // Remove mpt observedBy
    const auto& mpt = MapManager::Instance().GetMappoint(mptId);
    assert(mpt);
    assert(mpt->GetAnchoringKeyframeId() == id_);
    mpt->RemoveObservedByKeyframe(id_);

    // If this mappoint is not observed by other keyframe, no need to update the covisiblity graph
    if (mpt->GetObservedByKeyframeIds().size() == 0) {
        return;
    }

    // Need to update covisible graph if the mappoint are observed by other keyframes
    // Update covisible graph
    size_t newAnchorKFId = 0;
    for (auto& otherKFId: mpt->GetObservedByKeyframeIds()) {
        assert(otherKFId != this->id_);
        newAnchorKFId = otherKFId;

        auto otherKF = MapManager::Instance().GetKeyframe(otherKFId);
        assert(otherKF->IsObservingMappoint(mptId));

        auto& covisibleWeight = allCovisibleKfIdToWeight_[otherKFId];
        --covisibleWeight;
        if (covisibleWeight == 0) {
            allCovisibleKfIdToWeight_.erase(otherKFId);
        } else if (covisibleWeight < config_->activeCovisibleWeight) {
            activeCovisibleKfIds_.erase(otherKFId);
        }
        
        otherKF->UpdateCovisibleKeyframeWeight(this->id_, covisibleWeight);
    }
    // Need to update the anchor keyframe's id since the observation with the original one is removed
    mpt->AddAnchoringKeyframeId(newAnchorKFId);
}

#pragma mark - covisible keyframes

void Frame::UpdateCovisibleKeyframeWeight(const size_t otherKfId, const size_t weight) {
    if (weight == 0) {
        allCovisibleKfIdToWeight_.erase(otherKfId);
        activeCovisibleKfIds_.erase(otherKfId);
    } else if (weight >= config_->activeCovisibleWeight) {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        activeCovisibleKfIds_.insert(otherKfId);
    } else {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        activeCovisibleKfIds_.erase(otherKfId);
    }
}

}
