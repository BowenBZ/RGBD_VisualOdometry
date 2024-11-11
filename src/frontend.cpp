#include "myslam/frontend.h"

#include <cstddef>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>

#include "g2o/core/robust_kernel_impl.h"
#include "myslam/config.h"
#include "myslam/private/frame.h"
#include "myslam/private/g2o_types.h"
#include "myslam/private/mapmanager.h"

namespace myslam
{

Frontend::Frontend(const Camera::Ptr& camera) {

    camera_ = camera;

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
            [&](function<void(TrackingMap&)> updater) {
                UpdateTrackingMap(updater);
            });

    // Setup map manager
    mapManager_ = MapManager::Ptr(&MapManager::Instance());

    // Setup frame configs
    frameConfig_.maxFeaturesCnt = (size_t)Config::get<int>("frontend.number_of_features");
    frameConfig_.rowSectionCnt = (size_t)Config::get<int>("frontend.row_section_cnt");
    frameConfig_.colSectionCnt = (size_t)Config::get<int>("frontend.col_section_cnt");

    frameConfig_.imgCols = (size_t)Config::get<int>("frame.width");
    frameConfig_.imgRows = (size_t)Config::get<int>("frame.height");

    frameConfig_.gridSize = (size_t)Config::get<double>("pixel_grid_size");
    frameConfig_.gridColCnt = (size_t)ceil((double)frameConfig_.imgCols / frameConfig_.gridSize);
    frameConfig_.gridRowCnt = (size_t)ceil((double)frameConfig_.imgCols / frameConfig_.gridSize);

    frameConfig_.searchGridRadius = Config::get<int>("search_grid_radius");

    frameConfig_.descriptorDistanceThres = Config::get<double>("max_descriptor_distance");
    frameConfig_.bestSecondaryDistanceRatio = Config::get<double>("min_best_secondary_distance_ratio");

    frameConfig_.activeCovisibleWeight = (size_t)Config::get<double>("active_covisible_keyframe_weight");

    state_ = INITIALIZING;
}

bool Frontend::AddFrame(const Measurement& measurement)
{
    // Lock tracking map, so it cannot be updated during frontend processing
    unique_lock<mutex> lock(trackingMapMutex_);

    cout << "Frontend status: " << VOStateStr[state_] << endl;
    framePrev_ = frameCurr_;

    Frame::Ptr frame = Frame::CreateFrame(
            frameConfig_,
            measurement.timestamp,
            camera_,
            measurement.color,
            measurement.depth);
    frameCurr_ = frame;

    frameCurr_->ExtractKeyPointsAndComputeDescriptors(orb_);
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
        unordered_set<size_t> matchedKptsIdx;
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
    assert(lastFrameMpts_.size() == kptIdxToNewMpt_.size());
    for(auto& [kptIdx, mpt]: kptIdxToNewMpt_) {
        mapManager_->AddMappoint(mpt);
        frameCurr_->AddObservingMappointCreatedFromThisFrame(kptIdx, mpt);
    }
    trackingMap_.clear();
    trackingMap_.insert(lastFrameMpts_.begin(), lastFrameMpts_.end());

    // RGBD camera only needs 1 frame to configure since it could get the depth information
    state_ = TRACKING;
}

bool Frontend::TrackingHandler() {
    // Set an initial pose to the pose of previous pose, used for feature matching
    frameCurr_->SetTcw(framePrev_->GetTcw());

    // Compute pose based on last frame mappoints
    matchedKptIdxToInfo_.clear();
    printf("Frame tracking\n");
    MatchKeyPointsWithMappoints(lastFrameMpts_);
    EstimateCurrentFramePose(true);

    // Compute pose based on tracking map
    printf("Map tracking");
    MatchKeyPointsWithMappoints(trackingMap_);
    EstimateCurrentFramePose(true);

    // Create temp mappoints for next frame tracking
    CreateTempMappoints();

    if (!IsGoodEstimation()) {
        cout << "Cannot estimate Pose" << endl;
        accuLostFrameNums_++;
        state_ = (++accuLostFrameNums_ > frontendConfig_.maxLostFrames) ? LOST : TRACKING;
        return false;
    } 
    accuLostFrameNums_ = 0;
    
    if (IsKeyframe()) {
        printf("Current frame is a new keyframe\n");
        SendKeyframeToBackend();
    }

    return true;
}

void Frontend::LostHandler() {
    cout << "Tracking is lost" << endl;
}

void Frontend::UpdateTrackingMap(function<void(TrackingMap&)> updater) {
    unique_lock<mutex> lock(trackingMapMutex_);

    const SE3 old_T_kf_from_w = keyframeCurr_->GetTcw();

    // Use updater to update tracking map
    updater(trackingMap_);

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
        for (auto& [mptId, mpt]: lastFrameMpts_) {
            if (!trackingMap_.count(mptId)) {
                const Vector3d& oldPosition = mpt->GetPosition();
                const Vector3d newPosition = T * mpt->GetPosition();
                mpt->SetPosition(newPosition);
            }
        }
    }

    cout << "Tracking map is updated" << endl;
}

void Frontend::MatchKeyPointsWithMappoints(TrackingMap& trackingMap)
{
    Mat trackingMapDescriptors;
    unordered_map<int, size_t> trackingMapDescriptorIdxToMptId;
    unordered_map<size_t, size_t> matchedMptIdToKptIdx;

    for (auto &[mptId, mpt] : trackingMap)
    {
        trackingMapDescriptorIdxToMptId[trackingMapDescriptors.rows] = mptId;
        trackingMapDescriptors.push_back(mpt->GetDescriptor());
    }

    for (const auto& [kptIdx, matchInfo]: matchedKptIdxToInfo_) {
        matchedMptIdToKptIdx[matchInfo.mpt->GetId()] = kptIdx;
    }

    vector<cv::DMatch> matches;
    flannMatcher_.match(trackingMapDescriptors, frameCurr_->GetDescriptors(), matches);

    // compute the min distance of the best match
    float min_dis = std::min_element(
                        matches.begin(),
                        matches.end(),
                        [](const cv::DMatch &m1, const cv::DMatch &m2)
                        { return m1.distance < m2.distance; })
                        ->distance;
    float maxDis = max<float>(min_dis * frontendConfig_.minDisRatio, 30.0);

    for (cv::DMatch &m : matches)
    {
        // filter out the matches whose distance is large
        if (m.distance <= maxDis)
        {
            auto& mptId = trackingMapDescriptorIdxToMptId[m.queryIdx];
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

    cout << "  Size of tracking map: " << trackingMap.size() << endl;
    cout << "  Size of matched <keypoint, mappoint> pairs: " << matchedKptIdxToInfo_.size() << endl;
}

void Frontend::EstimateCurrentFramePose(const bool doMotionBA)
{
    // Construct the 3d-2d observations
    vector<size_t> kptIndices;
    vector<Point3f> pts3d;
    vector<Point2f> pts2d;

    for (auto& [kptIdx, info] : matchedKptIdxToInfo_) {
        kptIndices.push_back(kptIdx);
        pts3d.push_back(toPoint3f(info.mpt->GetPosition()));
        pts2d.push_back(frameCurr_->GetKeypoint(kptIdx).pt);
    }

    // Use P3P with RANSAC to compute the initial pose
    Mat initRotMat, rotVec, tranVec, inliers;
    cv::eigen2cv(frameCurr_->GetTcw().rotationMatrix(), initRotMat);
    cv::Rodrigues(initRotMat, rotVec);
    cv::eigen2cv(frameCurr_->GetTcw().translation(), tranVec);

    cv::solvePnPRansac(pts3d, pts2d, 
            camera_->GetCameraMatrix(), Mat(),
                    rotVec, tranVec, true,
                    100, 4.0, 0.99,
                        inliers, cv::SOLVEPNP_P3P);
    assert(inliers.rows != 0);
    printf("  Size of inlier after P3P ransac: %d\n", inliers.rows);

    // Covert rotation vector to matrix and to eigen types
    Mat rotMat;
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
    VertexPose *poseVertex = new VertexPose();
    poseVertex->setId(0);
    poseVertex->setEstimate(pnpEstimatedPose);
    optimizer_.addVertex(poseVertex);

    // Construct edges, optimizer.clear() will deallocate them
    vector<pair<UnaryEdgeProjection *, bool>> edges(inliers.rows);
    for (size_t inlierIdx = 0; inlierIdx < inliers.rows; ++inlierIdx)
    {
        int pointIdx = inliers.at<int>(inlierIdx);
        // 3D -> 2D projection
        UnaryEdgeProjection *edge = new UnaryEdgeProjection(toVector3d(pts3d[pointIdx]), camera_);
        edge->setId(inlierIdx);
        edge->setVertex(0, poseVertex);
        edge->setMeasurement(toVector2d(pts2d[pointIdx]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        // Each edge needs to have a separate kernel object,
        // optimizer.clear() will deallocate them
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
        rk->setDelta(sqrt(frontendConfig_.baInlierThres));
        edge->setRobustKernel(rk);

        // false means the edge is not outlier
        edges[inlierIdx] = std::make_pair(edge, false);
        optimizer_.addEdge(edge);
    }

    // Optimize 4 * 10 steps
    for (size_t iteration = 0; iteration < 4; ++iteration) {
        optimizer_.initializeOptimization(0);
        optimizer_.optimize(10);

        // Handle outlier edges
        for (auto& [edge, isOutlier]: edges) {
            // Compute error for outlier edges since they won't be calculated during optimization
            if (isOutlier) {
                edge->computeError();
            }

            // chi2 is the (u^2 + v^2)
            if (edge->chi2() > frontendConfig_.baInlierThres) {
                // level 1 edges won't be optimized later
                edge->setLevel(1);
                isOutlier = true;
            } else {
                edge->setLevel(0);
                isOutlier = false;
            }

            if (iteration == 2) {
                edge->setRobustKernel(nullptr);
            }
        }
    }

    // Update the inlier matched kpt -> mpt
    std::unordered_map<size_t, MatchInfo> baInlierKptIdxToInfo;
    for (size_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx)
    {
        if (edges[edgeIdx].second) {
            continue;
        }

        auto pointIdx = inliers.at<int>(edgeIdx);
        auto& kptIdx = kptIndices[pointIdx];
        baInlierKptIdxToInfo[kptIdx] = matchedKptIdxToInfo_[kptIdx];
    }
    printf("  Size of inlier after BA: %zu\n", baInlierKptIdxToInfo.size());
    matchedKptIdxToInfo_ = std::move(baInlierKptIdxToInfo);

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
        matchedTrackingMptCount += trackingMap_.count(matchInfo.mpt->GetId());
    }
    if (matchedTrackingMptCount < frontendConfig_.minInliersForKeyframe) {
        return true;
    }

    SE3 T_r_c = framePrev_->GetTcw() * frameCurr_->GetTcw().inverse();
    const Vector6d d = T_r_c.log();
    const Vector3d trans = d.head<3>();
    const Vector3d rot = d.tail<3>();
    const bool isLargeMotion = rot.norm() > frontendConfig_.maxFrameRotAllowed || trans.norm() > frontendConfig_.maxFrameTransAllowed;
    if (isLargeMotion) {
        return true;
    }
    return false;
}

void Frontend::CreateTempMappoints() {
    lastFrameMpts_.clear();
    kptIdxToNewMpt_.clear();
    for (size_t kptIdx = 0; kptIdx < frameCurr_->GetKeypointsSize(); ++kptIdx)
    {
        // If the keypoint matches with mappoint from local map, just put that mappoint into last frame mappoint
        if (matchedKptIdxToInfo_.count(kptIdx)) {
            const auto& mpt = matchedKptIdxToInfo_[kptIdx].mpt;
            if (trackingMap_.count(mpt->GetId())) {
                lastFrameMpts_[mpt->GetId()] = mpt;
                continue;
            }
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
        Mappoint::Ptr mpt = Mappoint::CreateMappoint(mptPos, frameCurr_->GetDescriptor(kptIdx));

        lastFrameMpts_[mpt->GetId()] = mpt;
        kptIdxToNewMpt_[kptIdx] = mpt;
    }
    // After creating temporary mappoints, the raw color and depth are no longer needed
    frameCurr_->ReleaseRawFrameData();
    printf("Created temp mappoints: %zu\n", kptIdxToNewMpt_.size());
}

void Frontend::SendKeyframeToBackend() {
    // Backend is idle when we could send new keyframe to backend

    // Firstly add this keyframe to map manager
    mapManager_->AddKeyframe(frameCurr_);
    keyframeCurr_ = frameCurr_;

    // Then add new keyframe's observation relationships for previous mappoints from local map
    for (const auto& [kptIdx, matchInfo] : matchedKptIdxToInfo_) {
        const auto& mpt = matchInfo.mpt;
        if (trackingMap_.count(mpt->GetId())) {
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