#include <myslam/frontend.hpp>

#include <myslam/config.hpp>

#include "myslam/private/mappoint.hpp"
#include "myslam/private/frame.hpp"
#include "myslam/private/superpoint_model.hpp"
#include "myslam/private/mapmanager.hpp"
#include "myslam/private/g2o_types.hpp"
#include "myslam/private/backend.hpp"
#include "myslam/private/util.hpp"

#include <opencv2/core/eigen.hpp>

#include <unordered_set>
#include <mutex>
#include <utility>

namespace myslam
{

Frontend::Frontend(const Camera::Ptr& camera): camera_(camera), mapManager_(&MapManager::Instance()) {
    // Setup superpoint model
    superpointModel_ = SuperPointModel::Ptr(new SuperPointModel(Config::get<std::string>("superpoint.path"), 
                                                                Config::get<double>("superpoint.confidenceThresh"),
                                                                Config::get<double>("superpoint.distThresh")));
    enableSuperpoint_ = Config::get<int>("superpoint.enable");
    nnThresh_ = Config::get<float>("superpoint.nnThresh");
    printf("SuperModel is initialized: %d\n", superpointModel_->Initialized());

    // Setup frontend config
    frontendConfig_.minDisRatio = Config::get<float>("frontend.match_ratio");
    frontendConfig_.baInlierThres = Config::get<double>("frontend.ba_inlier_threshold");
    frontendConfig_.minInliersForGood = (size_t)Config::get<int>("frontend.min_inliers_for_good_estimation");
    frontendConfig_.maxLostFrames = Config::get<float>("frontend.max_num_lost");
    frontendConfig_.minInliersForKeyframe = (size_t)Config::get<int>("frontend.min_inliers_for_new_keyframe");
    frontendConfig_.maxFrameRotAllowed = Config::get<double>("frontend.max_frame_rotation_allowed");
    frontendConfig_.maxFrameTransAllowed = Config::get<double>("frontend.max_frame_translation_allowed");

    // Feature detector and matcher
    orb_ = cv::ORB::create(Config::get<int>("frontend.number_of_features") / (Config::get<int>("frontend.row_section_cnt") * Config::get<int>("frontend.col_section_cnt")),
                            Config::get<double>("frontend.scale_factor"),
                            Config::get<int>("frontend.level_pyramid"));
    flannMatcher_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 10, 2));

    // Motion-only BA solver
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<DenseLinearSolverType>()));
    optimizer_.setAlgorithm(solver);

    // Setup backend
    backend_ = Backend::Ptr(new myslam::Backend(camera_));
    backend_->RegisterTrackingMapUpdateCallback(
            [&](std::function<void(TrackingMap&)> updater) {
                UpdateTrackingMap(updater);
            });

    // Setup frame configs
    frameConfig_ = std::shared_ptr<struct FrameConfig>(new struct FrameConfig);
    frameConfig_->maxFeaturesCnt = (size_t)Config::get<int>("frontend.number_of_features");
    frameConfig_->rowSectionCnt = (size_t)Config::get<int>("frontend.row_section_cnt");
    frameConfig_->colSectionCnt = (size_t)Config::get<int>("frontend.col_section_cnt");

    frameConfig_->imgCols = (size_t)Config::get<int>("frame.width");
    frameConfig_->imgRows = (size_t)Config::get<int>("frame.height");

    frameConfig_->gridSize = (size_t)Config::get<double>("pixel_grid_size");
    frameConfig_->gridColCnt = (size_t)ceil((double)frameConfig_->imgCols / frameConfig_->gridSize);
    frameConfig_->gridRowCnt = (size_t)ceil((double)frameConfig_->imgCols / frameConfig_->gridSize);

    frameConfig_->searchGridRadius = Config::get<int>("search_grid_radius");

    frameConfig_->descriptorDistanceThres = Config::get<double>("max_descriptor_distance");
    frameConfig_->bestSecondaryDistanceRatio = Config::get<double>("min_best_secondary_distance_ratio");

    frameConfig_->activeCovisibleWeight = (size_t)Config::get<double>("active_covisible_keyframe_weight");

    state_ = INITIALIZING;
}

bool Frontend::AddFrame(const Measurement& measurement)
{
    // Lock tracking map, so it cannot be updated during frontend processing
    std::unique_lock<std::mutex> lock(trackingMapMutex_);

    printf("Frontend status: %s\n", VOStateStr[state_].c_str());
    framePrev_ = frameCurr_;

    Frame::Ptr frame = Frame::CreateFrame(
            frameConfig_,
            measurement.timestamp,
            camera_,
            measurement.color,
            measurement.depth);
    frameCurr_ = frame;

    if (enableSuperpoint_) {
        frameCurr_->ExtractKeypointsAndDescriptorsWithSuperPointModel(superpointModel_);
    } else {
        frameCurr_->ExtractKeyPointsAndComputeDescriptors(orb_);
    }
    switch (state_)
    {
        case INITIALIZING:
        {
            InitializationHandler();
            break;
        }
        case TRACKING:
        {
            bool isTrackingGood = TrackingHandler();
            if (!isTrackingGood) {
                return false;
            }
            break;
        }
        case LOST:
        {   
            LostHandler();
            return false;
        }
    }

    if (viewer_)
    {
        std::unordered_set<size_t> matchedKptsIdx;
        for (const auto& match: matchedKptIdxToInfo_) {
            matchedKptsIdx.insert(match.first);
        }

        viewer_->SetCurrentFrame(measurement.color, frameCurr_, std::move(matchedKptsIdx));
        viewer_->UpdateDrawingObjects();
    }

    return true;
}

SE3 Frontend::GetPose() {
    return frameCurr_->GetTcw();
}

void Frontend::InitializationHandler() {
    // The first frame is a keyframe
    mapManager_->AddKeyframe(frameCurr_);
    keyframeCurr_ = frameCurr_;

    CreateTempMappoints();
    assert(lastFrameMap_.size() == kptIdxToNewMpt_.size());
    for(auto& [kptIdx, mpt]: kptIdxToNewMpt_) {
        mapManager_->AddMappoint(mpt);
        frameCurr_->AddObservingMappointCreatedFromThisFrame(kptIdx, mpt);
    }
    localMap_.clear();
    localMap_.insert(lastFrameMap_.begin(), lastFrameMap_.end());
    UpdateTrackingMapInfo(localMap_, localMapInfo_);

    // RGBD camera only needs 1 frame to configure since it could get the depth information
    state_ = TRACKING;
}

bool Frontend::TrackingHandler() {
    // Set an initial pose to the pose of previous pose, used for feature matching
    frameCurr_->SetTcw(framePrev_->GetTcw());

    // Compute pose based on last frame mappoints
    matchedKptIdxToInfo_.clear();
    if (enableSuperpoint_) {
        printf("Feature matching\n");
        MatchKeyPointsWithTrackingMapAndLastFrameNN();
        printf("Estimate pose\n");
        EstimateCurrentFramePose(true);
    } else {
        printf("Frame tracking\n");
        MatchKeyPointsWithMappoints(lastFrameMap_, lastFrameMapInfo_);
        EstimateCurrentFramePose(true);
    
        // Compute pose based on tracking map
        printf("Map tracking");
        MatchKeyPointsWithMappoints(localMap_, localMapInfo_);
        EstimateCurrentFramePose(true);
    }

    // Create temp mappoints for next frame tracking
    CreateTempMappoints();

    if (!IsGoodEstimation()) {
        printf("Cannot estimate Pose\n");
        accuLostFrameNums_++;
        state_ = (++accuLostFrameNums_ > frontendConfig_.maxLostFrames) ? LOST : TRACKING;
        return false;
    } 
    accuLostFrameNums_ = 0;
    
    if (IsKeyframe()) {
        SendKeyframeToBackend();
    }

    return true;
}

void Frontend::LostHandler() {
    printf("Tracking is lost\n");
}

void Frontend::UpdateTrackingMap(std::function<void(TrackingMap&)> updater) {
    std::unique_lock<std::mutex> lock(trackingMapMutex_);

    const SE3 old_T_kf_from_w = keyframeCurr_->GetTcw();

    // Use updater to update tracking map
    updater(localMap_);
    UpdateTrackingMapInfo(localMap_, localMapInfo_);

    // If the reference frame is different from keyframe, we also need to adjust the pose of the current frame
    if (frameCurr_ != keyframeCurr_) {
        const SE3 old_T_c_from_w = frameCurr_->GetTcw();
        const SE3 T_c_from_kf = old_T_c_from_w * old_T_kf_from_w.inverse();
        
        // Relative pose between frame and keyframe remains unchanged
        const SE3 new_T_kf_from_w = keyframeCurr_->GetTcw();
        const SE3 new_T_c_from_w = T_c_from_kf * new_T_kf_from_w;

        frameCurr_->SetTcw(new_T_c_from_w);

        const SE3 T = new_T_c_from_w.inverse() * old_T_c_from_w;
        // Update the mappoint position only observed by last frame
        for (auto& [mptId, mpt]: lastFrameMap_) {
            if (!localMap_.count(mptId)) {
                const Vector3d& oldPosition = mpt->GetPosition();
                const Vector3d newPosition = T * mpt->GetPosition();
                mpt->SetPosition(newPosition);
            }
        }
    }

    printf("Tracking map is updated\n");
}

void Frontend::UpdateTrackingMapInfo(const TrackingMap& trackingMap, TrackingMapInfo& info) {
    info.clear();

    for (auto &[mptId, mpt] : trackingMap)
    {
        info.descriptors.push_back(mpt->GetDescriptor());
        info.mptIds.push_back(mptId);
    }
}

#pragma mark - Feature matching

void Frontend::MatchKeyPointsWithMappoints(TrackingMap& trackingMap, TrackingMapInfo& info)
{
    std::unordered_map<size_t, size_t> matchedMptIdToKptIdx;
    for (const auto& [kptIdx, matchInfo]: matchedKptIdxToInfo_) {
        matchedMptIdToKptIdx[matchInfo.mpt->GetId()] = kptIdx;
    }

    std::vector<cv::DMatch> matches;
    flannMatcher_.match(info.descriptors, frameCurr_->GetDescriptors(), matches);

    // compute the min distance of the best match
    float min_dis = std::min_element(
                        matches.begin(),
                        matches.end(),
                        [](const cv::DMatch &m1, const cv::DMatch &m2)
                        { return m1.distance < m2.distance; })
                        ->distance;
    float maxDis = std::max<float>(min_dis * frontendConfig_.minDisRatio, 30.0);

    for (cv::DMatch &m : matches)
    {
        // filter out the matches whose distance is large
        if (m.distance <= maxDis)
        {
            auto& mptId = info.mptIds[m.queryIdx];
            auto& kptIdx = m.trainIdx;

            // Check whether this keypoint already has a better matched mappoint
            if (matchedKptIdxToInfo_.count(kptIdx) && 
                m.distance >= matchedKptIdxToInfo_[kptIdx].distance) {
                continue;
            }

            // Check if this mappoint already matches with other keypoint
            if (matchedMptIdToKptIdx.count(mptId)) {
                const size_t previousMatchedKpt = matchedMptIdToKptIdx[mptId];
                if (m.distance >= matchedKptIdxToInfo_[previousMatchedKpt].distance) {
                    continue;
                }
                matchedMptIdToKptIdx[mptId] = kptIdx;
            }

            matchedKptIdxToInfo_[kptIdx] = {trackingMap[mptId], m.distance};
        }
    }

    printf("  Size of tracking map: %zu\n", trackingMap.size());
    printf("  Size of matched <keypoint, mappoint> pairs: %zu\n", matchedKptIdxToInfo_.size());
}

void Frontend::FindMatch(TrackingMap& trackingMap, TrackingMapInfo& trackingMapInfo, cv::Mat currDescriptors, const float nnThresh, std::unordered_map<size_t, MatchInfo>& matchedKptIdxToInfo) {

    KeypointsMatchInfo matchInfo;
    find_matched_points(trackingMapInfo.descriptors, currDescriptors, nnThresh, matchInfo);
    assert(matchInfo.matchedCurrentIndices.size() == matchInfo.matchedPrevIndices.size());

    // Take all matches with map, and fill in remained one from last frame
    for (int i = 0; i < matchInfo.matchedCurrentIndices.size(); i++) {
        const int currentIdx = matchInfo.matchedCurrentIndices[i];
        const int prevIdx = matchInfo.matchedPrevIndices[i];
        const int matchedMptId = trackingMapInfo.mptIds[prevIdx];

        // This kpt already has matches or this mappoint already has matches
        if (matchedKptIdxToInfo.count(currentIdx)) {
            continue;
        }

        const float distance = matchInfo.matchedDistance[i];        
        matchedKptIdxToInfo[currentIdx] = {trackingMap[matchedMptId], distance};
    }
}

void Frontend::MatchKeyPointsWithTrackingMapAndLastFrameNN() {
    matchedKptIdxToInfo_.clear();

    // Try to match with other mappoints from local map
    FindMatch(localMap_, localMapInfo_, frameCurr_->GetDescriptors(), nnThresh_, matchedKptIdxToInfo_);
    const size_t trackingMapMatchedCnt = matchedKptIdxToInfo_.size();
    
    // Try to match with new created temporay mappoints
    FindMatch(lastFrameMap_, lastFrameMapInfo_, frameCurr_->GetDescriptors(), nnThresh_, matchedKptIdxToInfo_);
    const size_t totalMatchedCnt = matchedKptIdxToInfo_.size();
    const size_t tempMatchedCnt = totalMatchedCnt - trackingMapMatchedCnt;

    printf("  Size of matched <keypoint, mappoint> pairs: %zu (%zu + %zu)\n", 
                totalMatchedCnt, trackingMapMatchedCnt, tempMatchedCnt);
}

#pragma mark - Motion-only BA

void Frontend::EstimateCurrentFramePose(const bool doMotionBA)
{
    // Construct the 3d-2d observations
    std::vector<size_t> kptIndices;
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (auto& [kptIdx, info] : matchedKptIdxToInfo_) {
        kptIndices.push_back(kptIdx);
        pts3d.push_back(toPoint3f(info.mpt->GetPosition()));
        pts2d.push_back(frameCurr_->GetKeypoint(kptIdx).pt);
    }

    // Use P3P with RANSAC to compute the initial pose
    cv::Mat initRotMat, rotVec, tranVec, inliers;
    cv::eigen2cv(frameCurr_->GetTcw().rotationMatrix(), initRotMat);
    cv::Rodrigues(initRotMat, rotVec);
    cv::eigen2cv(frameCurr_->GetTcw().translation(), tranVec);

    cv::solvePnPRansac(pts3d, pts2d, 
            camera_->GetCameraMatrix(), cv::Mat(),
                    rotVec, tranVec, true,
                    100, 4.0, 0.99,
                        inliers, cv::SOLVEPNP_P3P);
    assert(inliers.rows != 0);
    printf("  Size of inlier after P3P ransac: %d\n", inliers.rows);

    // Covert rotation std::vector to matrix and to eigen types
    cv::Mat rotMat;
    cv::Rodrigues(rotVec, rotMat);
    Eigen::Matrix3d rotMatEigen;
    Vector3d tranVecEigen;
    cv::cv2eigen(rotMat, rotMatEigen);
    cv::cv2eigen(tranVec, tranVecEigen);
    SE3 pnpEstimatedPose = SE3(rotMatEigen, tranVecEigen);

    if (!doMotionBA) {
        frameCurr_->SetTcw(pnpEstimatedPose);
        return;
    }

    // Construct pose vertex
    // No need to manually release the memory since the memory will be released when calling optimizer.clear()
    VertexPose* poseVertex = new VertexPose();
    poseVertex->setId(0);
    poseVertex->setEstimate(pnpEstimatedPose);
    optimizer_.addVertex(poseVertex);

    // Construct edges, optimizer.clear() will deallocate them
    std::list<EdgeInfo> edgesInfo;
    for (size_t idx = 0; idx < inliers.rows; idx++)
    {
        const size_t inlierIdx = inliers.at<int>(idx);
        const size_t kptIdx = kptIndices[inlierIdx];
        const auto mpt = matchedKptIdxToInfo_[kptIdx].mpt;

        // 3D -> 2D projection
        UnaryEdgeProjection *edge = new UnaryEdgeProjection(mpt->GetPosition(), camera_);
        edge->setId(idx);
        edge->setVertex(0, poseVertex);
        edge->setMeasurement(toVector2d(frameCurr_->GetKeypoint(kptIdx).pt));
        edge->setInformation(Eigen::Matrix2d::Identity());
        // Each edge needs to have a separate kernel object,
        // optimizer.clear() will deallocate them
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
        rk->setDelta(sqrt(frontendConfig_.baInlierThres));
        edge->setRobustKernel(rk);

        // false means the edge is not outlier
        edgesInfo.push_back({edge, false, kptIdx});
        optimizer_.addEdge(edge);
    }

    // Optimize 4 * 10 steps
    for (size_t iteration = 0; iteration < 4; ++iteration) {
        optimizer_.initializeOptimization(0);
        optimizer_.optimize(10);

        // Handle outlier edges
        for (auto& edgeInfo: edgesInfo) {
            auto& edge = edgeInfo.edge;
            // Compute error for outlier edges since they won't be calculated during optimization
            if (edgeInfo.isOutlier) {
                edge->computeError();
            }

            // chi2 is the (u^2 + v^2)
            if (edge->chi2() > frontendConfig_.baInlierThres) {
                // level 1 edges won't be optimized later
                edge->setLevel(1);
                edgeInfo.isOutlier = true;
            } else {
                edge->setLevel(0);
                edgeInfo.isOutlier = false;
            }

            if (iteration == 2) {
                edge->setRobustKernel(nullptr);
            }
        }
    }

    // Only keep the inlier matched kpt -> mpt
    for (auto& edgeInfo: edgesInfo) {
        if (edgeInfo.isOutlier) {
            matchedKptIdxToInfo_.erase(edgeInfo.kptIdx);
        }
    }
    printf("  Size of inlier after BA: %zu\n", matchedKptIdxToInfo_.size());

    // Set computed pose
    frameCurr_->SetTcw(poseVertex->estimate());

    // Clear allocated vertex and edge,
    // also deallocates the memory associated with them.
    optimizer_.clear();
}

bool Frontend::IsGoodEstimation()
{
    // check if inliers number meet the threshold
    if (matchedKptIdxToInfo_.size() < frontendConfig_.minInliersForGood) {
        printf("Current tracking is rejected because inlier is too small: %zu\n", matchedKptIdxToInfo_.size());
        return false;
    }

    // check if the motion is too large
    SE3 T_r_c = framePrev_->GetTcw() * frameCurr_->GetTcw().inverse();
    float d = T_r_c.log().norm();
    if (d > 5.0) {
        printf("Current tracking is rejected because motion is too large: %f\n", d);
        return false;
    }
    return true;
}

bool Frontend::IsKeyframe()
{
    if (!backend_->IsIdle()) {
        return false;
    }

    size_t matchedTrackingMptCount = 0;
    for (const auto& [kpt, matchInfo]: matchedKptIdxToInfo_) {
        matchedTrackingMptCount += localMap_.count(matchInfo.mpt->GetId());
    }
    if (matchedTrackingMptCount < frontendConfig_.minInliersForKeyframe) {
        printf("Current frame is a new keyframe since matched mpt count %zu < %zu\n", matchedTrackingMptCount, frontendConfig_.minInliersForKeyframe);
        return true;
    }

    SE3 T_r_c = framePrev_->GetTcw() * frameCurr_->GetTcw().inverse();
    const Eigen::Matrix<double, 6, 1> d = T_r_c.log();
    const Vector3d trans = d.head<3>();
    const Vector3d rot = d.tail<3>();
    const bool isLargeMotion = rot.norm() > frontendConfig_.maxFrameRotAllowed || trans.norm() > frontendConfig_.maxFrameTransAllowed;
    if (isLargeMotion) {
        printf("Current frame is a new keyframe since motion is too large\n");
        return true;
    }
    return false;
}

void Frontend::CreateTempMappoints() {
    lastFrameMap_.clear();
    kptIdxToNewMpt_.clear();
    for (size_t kptIdx = 0; kptIdx < frameCurr_->GetKeypointsSize(); ++kptIdx)
    {
        // If the keypoint matches with mappoint from local map, just put that mappoint into last frame mappoint
        if (matchedKptIdxToInfo_.count(kptIdx)) {
            const auto& mpt = matchedKptIdxToInfo_[kptIdx].mpt;
            if (!localMap_.count(mpt->GetId())) {
                // if it's not from local map
                lastFrameMap_[mpt->GetId()] = mpt;
            }
            continue;
        }

        // Check if the keypoint has depth value
        auto& kpt = frameCurr_->GetKeypoint(kptIdx);
        double depth = frameCurr_->GetDepth(kpt);
        if (depth < 0) {
            continue;
        }
        // TODO: check the depth is in a reasonable region

        Vector3d mptPos = camera_->Pixel2World(
            kpt, frameCurr_->GetTcw(), depth);
        
        // Create a mappoint
        // all parameters will have a deep copy inside the constructor
        Mappoint::Ptr mpt = Mappoint::CreateMappoint(mptPos, frameCurr_->GetDescriptor(kptIdx), enableSuperpoint_);

        lastFrameMap_[mpt->GetId()] = mpt;
        kptIdxToNewMpt_[kptIdx] = mpt;
    }
    UpdateTrackingMapInfo(lastFrameMap_, lastFrameMapInfo_);

    // After creating temporary mappoints, the raw color and depth are no longer needed
    frameCurr_->ReleaseRawFrameData();
    printf("Created %zu new temp mappoints, total mappoints from this frame: %zu\n", 
            kptIdxToNewMpt_.size(), lastFrameMap_.size());
}

void Frontend::SendKeyframeToBackend() {
    // Backend is idle when we could send new keyframe to backend

    // Firstly add this keyframe to map manager
    mapManager_->AddKeyframe(frameCurr_);
    keyframeCurr_ = frameCurr_;

    // Then add new keyframe's observation relationships for previous mappoints from local map
    for (const auto& [kptIdx, matchInfo] : matchedKptIdxToInfo_) {
        const auto& mpt = matchInfo.mpt;
        if (localMap_.count(mpt->GetId())) {
            frameCurr_->AddObservingMappointCreatedFromOtherFrame(kptIdx, mpt);
        }
    }

    // Then add the new mappoints to map manager, and new keyframe's observation relationship
    for (const auto& [kptIdx, mpt ]: kptIdxToNewMpt_) {
        mapManager_->AddMappoint(mpt);
        frameCurr_->AddObservingMappointCreatedFromThisFrame(kptIdx, mpt);
    }

    // Add keyframe id to backend's queue
    backend_->AddNewKeyframeInfoToQueue(frameCurr_->GetId());
}

void Frontend::Stop() {
    backend_->Stop();
    if (viewer_ != nullptr) {
        viewer_->Stop();
    }
}

} // namespace