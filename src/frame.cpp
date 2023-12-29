/*
 * Class frame maintains its RGB and depth image, and its pose in world reference frame
 *
 * When the instance is regarded as a keyframe, it will populate the observed mappoints and covisible keyframes, 
 * where the active keyframes share at least 15 covisible mappoints of this keyframe
 * 
 * After adding/removing an observed mappoints, the covisible keyframes will be updated automatcially
 */

#include "myslam/frame.h"
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
        ++factoryId_,
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
    imgCols = color.cols;
    imgRows = color.rows;

    gridSize_ = (size_t)Config::get<double>("pixel_grid_size");
    gridColCnt_ = (size_t)ceil((double)imgCols / gridSize_);
    gridRowCnt_ = (size_t)ceil((double)imgRows / gridSize_);

    searchGridRadius_ = Config::get<int>("search_grid_radius");

    descriptorDistanceThres_ = Config::get<double>("max_descriptor_distance");
    bestSecondaryDistanceRatio_ = Config::get<double>("min_best_secondary_distance_ratio");;
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
    detector->detectAndCompute(color_, Mat(), keypoints_, descriptors_);
    ConstructKeypointGrids();

    // TODO: Since the keypoints are extracted, the color frame won't been needed unless the viewer
}

void Frame::ConstructKeypointGrids() {
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        size_t gridIdx = GetGridIdx(keypoints_[i].pt.x, keypoints_[i].pt.y);
        gridToKptIdx_[gridIdx].push_back(i);
    }
}

bool Frame::GetMatchedKeypoint(const Mappoint::Ptr& mpt, size_t& kptIdx, double& distance, bool& mayObserveMpt) {
    mayObserveMpt = false;

    Vector3d posInCam = camera_->World2Camera(mpt->GetPosition(), T_c_w_);
    if (posInCam(2, 0) < 0) {
        return false;
    } 

    Vector2d pixelPos = camera_->Camera2Pixel(posInCam);
    if (pixelPos(0, 0) < 0 || pixelPos(0, 0) >= imgCols ||
        pixelPos(1, 0) < 0 || pixelPos(1, 0) >= imgRows) {
        return false;
    }

    Vector3d direction = mpt->GetPosition() - this->GetCamCenter();
    direction.normalize();
    double angle = acos( direction.transpose() * mpt->GetNormDirection() );
    if ( angle > M_PI / 6 ) {
        return false;
    }

    // This frame may observe this mappoint, but not gurantee to have a matched keypoint
    // This is used by the fallback flann feature matching
    mayObserveMpt = true;
    
    const size_t mptGridIdx = GetGridIdx(pixelPos(0, 0), pixelPos(0, 1));
    const vector<size_t> queryGridsIdx = getNearbyGrids(mptGridIdx);
    vector<pair<size_t, double>> kptIdxToDistance;
    for (auto& gridIdx: queryGridsIdx) {
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
    // if (bestKptToDistance.second > descriptorDistanceThres_) {
    //     return false;
    // }

    // if (kptIdxToDistance.size() >= 2) {
    //     const pair<size_t, double>& secondKptToDistance = kptIdxToDistance[1];
    //     if (bestKptToDistance.second / secondKptToDistance.second < bestSecondaryDistanceRatio_) {
    //         return false;
    //     }
    // }
    
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

vector<size_t> Frame::getNearbyGrids(size_t gridIdx) {
    vector<size_t> grids;

    size_t rowIdx = gridIdx / gridColCnt_;
    size_t colIdx = gridIdx - rowIdx * gridColCnt_;

    for (int drow = -searchGridRadius_; drow <= searchGridRadius_; ++drow) {
        for (int dcol = -searchGridRadius_; dcol <= searchGridRadius_; ++dcol) {
            int row = (int)rowIdx + drow;
            int col = (int)colIdx + dcol;

            if (row >= 0 && row < gridRowCnt_ &&
                col >= 0 && col < gridColCnt_) {
                    grids.push_back(GetGridIdx((size_t)row, (size_t)col));
                }
        }
    }

    return grids;
}

void Frame::AddObservingMappoint(const Mappoint::Ptr& mpt, const size_t kptIdx) {
    unique_lock<mutex> lck(observationMutex_);
    
    auto mptId = mpt->GetId();
    assert(!observingMappointIds_.count(mptId));
    observingMappointIds_.insert(mptId);

    assert(mpt != nullptr);
    mpt->AddObservedByKeyframe(shared_from_this(), kptIdx);

    for (auto& [otherKFId, _]: mpt->GetObservedByKeyframesMap()) {
        if (otherKFId == id_) {
            continue;
        }

        auto otherKF = MapManager::GetInstance().GetKeyframe(otherKFId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservingMappoint(mptId));

        ++allCovisibleKeyframeIdToWeight_[otherKFId];
        if (allCovisibleKeyframeIdToWeight_[otherKFId] >= 15) {
            activeCovisibleKeyframes_.insert(otherKFId);
        }

        otherKF->UpdateCovisibleKeyframeWeight(id_, allCovisibleKeyframeIdToWeight_[otherKFId]);
    }
}

void Frame::RemoveObservedMappoint(const size_t mappointId) {
    unique_lock<mutex> lck(observationMutex_);

    assert(observingMappointIds_.count(mappointId));
    observingMappointIds_.erase(mappointId);

    auto mappoint = MapManager::GetInstance().GetMappoint(mappointId);
    assert(mappoint != nullptr);
    mappoint->RemoveObservedByKeyframe(this->id_);

    for (auto& [otherKFId, _]: mappoint->GetObservedByKeyframesMap()) {
        if (otherKFId == this->id_) {
            continue;
        }

        auto otherKF = MapManager::GetInstance().GetKeyframe(otherKFId);
        assert(otherKF != nullptr);
        assert(otherKF->IsObservingMappoint(mappointId));

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
