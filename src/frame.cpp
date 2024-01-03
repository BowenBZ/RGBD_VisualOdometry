#include "myslam/frame.h"

#include <algorithm>

#include "myslam/util.h"
#include "myslam/mapmanager.h"

namespace myslam
{

size_t Frame::factoryId_ = 0;

Frame::Ptr Frame::CreateFrame(
    const double timestamp, 
    const Camera::Ptr& camera, 
    const Mat& color, 
    const Mat& depth)
{
    return Frame::Ptr( new Frame(
        factoryId_++,
        timestamp,
        camera,
        color,
        depth) 
    );
}

Frame::Frame (  const size_t id, 
                const double timestamp, 
                const Camera::Ptr& camera, 
                const Mat& color, 
                const Mat& depth )
: id_(id), timestamp_(timestamp), camera_(camera), color_(color.clone()), depth_(depth.clone()), T_c_w_(SE3())
{
    maxFeaturesCnt_ = (size_t)Config::get<int>("number_of_features");
    rowSectionCnt_ = (size_t)Config::get<int>("row_section_cnt");
    colSectionCnt_ = (size_t)Config::get<int>("col_section_cnt");

    imgCols_ = color.cols;
    imgRows_ = color.rows;

    gridSize_ = (size_t)Config::get<double>("pixel_grid_size");
    gridColCnt_ = (size_t)ceil((double)imgCols_ / gridSize_);
    gridRowCnt_ = (size_t)ceil((double)imgRows_ / gridSize_);

    searchGridRadius_ = Config::get<int>("search_grid_radius");

    descriptorDistanceThres_ = Config::get<double>("max_descriptor_distance");
    bestSecondaryDistanceRatio_ = Config::get<double>("min_best_secondary_distance_ratio");

    activeCovisibleWeight_ = (size_t)Config::get<double>("active_covisible_keyframe_weight");
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

void Frame::ExtractKeyPointsAndComputeDescriptors(const cv::Ptr<cv::Feature2D>& detector) {
    
    size_t rowPerSection = imgRows_ / rowSectionCnt_;
    size_t colPerSection = imgCols_ / colSectionCnt_;
    for(size_t rowSection = 0; rowSection < rowSectionCnt_; ++rowSection) {
        for (size_t colSection = 0; colSection < colSectionCnt_; ++colSection) {
            size_t rowStartIdx = rowPerSection * rowSection;
            size_t rowEndIdx = rowPerSection * (rowSection + 1);
            size_t colStartIdx = colPerSection * colSection;
            size_t colEndIdx = colPerSection * (colSection + 1);

            cv::Range rowRange(rowStartIdx, rowEndIdx);
            cv::Range colRange(colStartIdx, colEndIdx);

            vector<KeyPoint> kpts;
            Mat des;
            detector->detectAndCompute(color_(rowRange, colRange), Mat(), kpts, des);

            for (size_t idx = 0; idx < min(kpts.size(), maxFeaturesCnt_ / (rowSectionCnt_ * colSectionCnt_)); ++idx) {
                auto& kpt = kpts[idx];
                kpt.pt.x += colStartIdx;
                kpt.pt.y += rowStartIdx;
                keypoints_.push_back(kpt);
                descriptors_.push_back(des.row(idx).clone());
            }
        }
    }
    
    ConstructKeypointGrids();

    // TODO: Since the keypoints are extracted, the color frame won't been needed unless the viewer
    // TODO: Extract more keypoints for keyframe
}

void Frame::ConstructKeypointGrids() {
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        size_t gridIdx = GetGridIdx(keypoints_[i].pt.x, keypoints_[i].pt.y);
        gridToKptIdx_[gridIdx].push_back(i);
    }
}

bool Frame::GetMatchedKeypoint(const Mappoint::Ptr& mpt, const bool doDirectionCheck, size_t& kptIdx, double& distance, bool& mayObserveMpt) {
    mayObserveMpt = false;

    Vector3d posInCam = camera_->World2Camera(mpt->GetPosition(), T_c_w_);
    if (posInCam[2] < 0) {
        return false;
    } 
    
    Vector2d pixelPos = camera_->Camera2Pixel(posInCam);
    if (pixelPos[0] < 0 || pixelPos[0] >= imgCols_ ||
        pixelPos[1] < 0 || pixelPos[1] >= imgRows_) {
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
    list<size_t> nearbyGrids;
    getNearbyGrids(mptGridIdx, nearbyGrids);
    vector<pair<size_t, double>> kptIdxToDistance;
    for (auto& gridIdx: nearbyGrids) {
        if (!gridToKptIdx_.count(gridIdx)) {
            continue;
        }

        for (auto& kptIdx: gridToKptIdx_[gridIdx]) {
            double distance = ComputeDescriptorDistance(
                mpt->GetDescriptor(), 0,
                descriptors_, kptIdx);

            kptIdxToDistance.push_back({kptIdx, distance});
        }
    }

    if (kptIdxToDistance.empty()) {
        return false;
    }

    sort(kptIdxToDistance.begin(), kptIdxToDistance.end(), 
        [](const pair<size_t, double>& kpt1, const pair<size_t, double>& kpt2) {
            return kpt1.second < kpt2.second;
        });

    const pair<size_t, double>& bestKptToDistance = kptIdxToDistance[0];
    if (bestKptToDistance.second > descriptorDistanceThres_) {
        return false;
    }

    if (kptIdxToDistance.size() >= 2) {
        const pair<size_t, double>& secondKptToDistance = kptIdxToDistance[1];
        if (bestKptToDistance.second / secondKptToDistance.second < bestSecondaryDistanceRatio_) {
            return false;
        }
    }
    
    kptIdx = bestKptToDistance.first;
    distance = bestKptToDistance.second;
    return true;
}

size_t Frame::GetGridIdx(double x, double y) {
    size_t colIdx = (size_t)floor(x / gridSize_);
    size_t rowIdx = (size_t)floor(y / gridSize_);
    return GetGridIdx(colIdx, rowIdx);
}

size_t Frame::GetGridIdx(size_t colIdx, size_t rowIdx) {
    return rowIdx * gridColCnt_ + colIdx;
}

void Frame::getNearbyGrids(size_t gridIdx, list<size_t>& nearbyGrids) {
    nearbyGrids.clear();

    size_t rowIdx = gridIdx / gridColCnt_;
    size_t colIdx = gridIdx - rowIdx * gridColCnt_;

    for (int drow = -searchGridRadius_; drow <= searchGridRadius_; ++drow) {
        for (int dcol = -searchGridRadius_; dcol <= searchGridRadius_; ++dcol) {
            int row = (int)rowIdx + drow;
            int col = (int)colIdx + dcol;

            if (row >= 0 && row < gridRowCnt_ &&
                col >= 0 && col < gridColCnt_) {
                    nearbyGrids.push_back(GetGridIdx((size_t)row, (size_t)col));
                }
        }
    }
}

void Frame::AddObservingMappoint(const Mappoint::Ptr& mpt, const size_t kptIdx) {
    unique_lock<mutex> lck(observationMutex_);
    
    auto mptId = mpt->GetId();
    assert(!observingMptIdToKptIdxMap_.count(mptId));
    observingMptIdToKptIdxMap_[mptId] = kptIdx;
    kptIdxToObservingMptIdMap_[kptIdx] = mptId;

    assert(mpt != nullptr);
    mpt->AddObservedByKeyframe(shared_from_this(), kptIdx);

    unordered_map<size_t, size_t> observedByKfIdToKptIdx;
    mpt->GetObservedByKeyframesMap(observedByKfIdToKptIdx);
    for (auto& [otherKfId, _]: observedByKfIdToKptIdx) {
        if (otherKfId == id_) {
            continue;
        }

        auto otherKF = MapManager::Instance().GetKeyframe(otherKfId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservingMappoint(mptId));

        ++allCovisibleKfIdToWeight_[otherKfId];
        allCovisibleKfIds_.insert(otherKfId);
        if (allCovisibleKfIdToWeight_[otherKfId] >= activeCovisibleWeight_) {
            activeCovisibleKfIds_.insert(otherKfId);
        }

        otherKF->UpdateCovisibleKeyframeWeight(id_, allCovisibleKfIdToWeight_[otherKfId]);
    }
}

void Frame::RemoveObservingMappoint(const size_t mptId) {
    unique_lock<mutex> lck(observationMutex_);

    assert(observingMptIdToKptIdxMap_.count(mptId));
    size_t kptIdx = observingMptIdToKptIdxMap_[mptId];
    observingMptIdToKptIdxMap_.erase(mptId);
    kptIdxToObservingMptIdMap_.erase(kptIdx);

    auto mpt = MapManager::Instance().GetMappoint(mptId);
    assert(mpt != nullptr);
    mpt->RemoveObservedByKeyframe(this->id_);

    unordered_map<size_t, size_t> observedByKfIdToKptIdx;
    mpt->GetObservedByKeyframesMap(observedByKfIdToKptIdx);
    for (auto& [otherKFId, _]: observedByKfIdToKptIdx) {
        if (otherKFId == this->id_) {
            continue;
        }

        auto otherKF = MapManager::Instance().GetKeyframe(otherKFId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservingMappoint(mptId));

        --allCovisibleKfIdToWeight_[otherKFId];
        size_t newWeight = allCovisibleKfIdToWeight_[otherKFId];
        if (newWeight == 0) {
            allCovisibleKfIdToWeight_.erase(otherKFId);
            allCovisibleKfIds_.erase(otherKFId);
        } else if (newWeight < activeCovisibleWeight_) {
            activeCovisibleKfIds_.erase(otherKFId);
        }
        
        otherKF->UpdateCovisibleKeyframeWeight(this->id_, newWeight);
    }

    // TODO: if all the observations has been removed, consider this keyframe as outlier?
}


void Frame::UpdateCovisibleKeyframeWeight(const size_t otherKfId, const size_t weight) {
    unique_lock<mutex> lck(observationMutex_);

    if (weight == 0) {
        allCovisibleKfIdToWeight_.erase(otherKfId);
        allCovisibleKfIds_.erase(otherKfId);
        activeCovisibleKfIds_.erase(otherKfId);
    } else if (weight >= activeCovisibleWeight_) {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        allCovisibleKfIds_.insert(otherKfId);
        activeCovisibleKfIds_.insert(otherKfId);
    } else {
        allCovisibleKfIdToWeight_[otherKfId] = weight;
        allCovisibleKfIds_.insert(otherKfId);
        activeCovisibleKfIds_.erase(otherKfId);
    }
}

}
