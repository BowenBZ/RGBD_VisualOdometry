/*
 * Fontend which tracks camera poses on frames
 *
 * The entry point is the AddFrame function, which gets a frame pointer and computes the camera pose on that frame. Return false if the tracking fails.
 *
 * Frontend is also reponsible for the other functions
 * 1. create keyframe
 * 2. create new mappoints for the keyframe
 * 3. maintain the covisible graph of keyframes (consider in backend??)
 * 4. invoke backend (if there is) to optimize
 * 5. invoke reviewer (if there is) to show the image frames, real-time poses and maps
 */

#include "myslam/frontend.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>

#include "myslam/config.h"
#include "myslam/g2o_types.h"
#include "myslam/mapmanager.h"

namespace myslam
{

Frontend::Frontend()    
{
    state_ = INITIALIZING;

    flannMatcher_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 10, 2));

    orb_ = cv::ORB::create(Config::get<int>("number_of_features"),
                            Config::get<double>("scale_factor"),
                            Config::get<int>("level_pyramid"));
    minMatchesToUseFlannFrameTracking_ = (size_t)Config::get<double>("min_matches_to_use_flann_frame_tracking");
    minMatchesToUseFlannMapTracking_ = (size_t)Config::get<double>("min_matches_to_use_flann_map_tracking");
    minDisRatio_ = Config::get<float>("match_ratio");
    baInlierThres_ = Config::get<double>("ba_inlier_threshold");
    minInliersForGood_ = (size_t)Config::get<int>("min_inliers_for_good_estimation");
    maxLostFrames_ = Config::get<float>("max_num_lost");
    minInliersForKeyframe_ = (size_t)Config::get<int>("min_inliers_for_new_keyframe");
    keyFrameMinRot_ = Config::get<double>("keyframe_rotation");
    keyFrameMinTrans_ = Config::get<double>("keyframe_translation");

    // using motion-only bundle adjustment to optimize the pose
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<DenseLinearSolverType>()));
    optimizer_.setAlgorithm(solver);
}

bool Frontend::AddFrame(const Frame::Ptr frame)
{
    cout << "Frontend status: " << VOStateStr[state_] << endl;
    framePrev_ = frameCurr_;
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
        for (const auto& [kptIdx, _]: matchedKptIdxMptIdMap_) {
            matchedKptsIdx.insert(kptIdx);
        }

        viewer_->SetCurrentFrame(frameCurr_, matchedKptsIdx, baInlierKptIdxSet_);
        viewer_->UpdateDrawingObjects();
    }

    return true;
}

void Frontend::InitializationHandler() {
    // the first frame is a keyframe
    MapManager::GetInstance().InsertKeyframe(frameCurr_);
    CreateTempMappoints();
    AddTempMappointsToMapManager();
    AddObservingMappointsToCurrentFrame();
    UpdateTrackingMap(frameCurr_, nullptr);

    // RGBD camera only needs 1 frame to configure since it could get the depth information
    state_ = TRACKING;
}

bool Frontend::TrackingHandler() {
    // set an initial pose to the pose of previous pose, used for feature matching
    frameCurr_->SetPose(framePrev_->GetPose());

    // lock tracking map
    unique_lock<mutex> lock(trackingMapMutex_);

    // Compute pose based on last frame mappoints
    cout << "Frame tracking...\n";
    MatchKeyPointsWithMappoints(lastFrameMpts_, minMatchesToUseFlannFrameTracking_);
    EstimatePoseMotionOnlyBA(lastFrameMpts_);

    // Compute pose based on tracking map
    cout << "Map tracking...\n";
    MatchKeyPointsWithMappoints(trackingMap_, minMatchesToUseFlannMapTracking_);
    EstimatePoseMotionOnlyBA(trackingMap_);

    // Create temp mappoints for next frame tracking
    CreateTempMappoints();

    if (!IsGoodEstimation())
    {
        cout << "Cannot estimate Pose" << endl;
        accuLostFrameNums_++;
        state_ = (++accuLostFrameNums_ > maxLostFrames_) ? LOST : TRACKING;
        return false;
    } else {
        accuLostFrameNums_ = 0;
    }
    
    if (!IsKeyframe()) {
        return true;
    } else {
        cout << "Current frame is a new keyframe" << endl;
    }

    MapManager::GetInstance().InsertKeyframe(frameCurr_);
    AddTempMappointsToMapManager();
    AddObservingMappointsToCurrentFrame();

    // AddNewMappointsObservationsForOldKeyframes();
    
    // if have backend, use backend to optimize mappoints position and frame pose
    if (backend_) {
        // unlock tracking map, otherwise it may cause deadly lock
        lock.unlock();
        // TODO: if backend is optimizaing, here will block frontend
        backend_->OptimizeCovisibleGraphOfKeyframe(frameCurr_);
    } else {
        TriangulateMappointsInTrackingMap();
    }

    return true;
}

void Frontend::LostHandler() {
    cout << "Tracking is lost" << endl;
}

void Frontend::UpdateTrackingMap(const Frame::Ptr& keyframe, function<void(void)> callback) {
    unique_lock<mutex> lock(trackingMapMutex_);

    // Tracking map is defined by reference keyframe
    if (keyframeForTrackingMap_ == nullptr || keyframe->GetId() != keyframeForTrackingMap_->GetId()) {
        keyframeForTrackingMap_ = keyframe;
        trackingMap_ = MapManager::GetInstance().GetMappointsAroundKeyframe(keyframe);

        if (trackingMap_.size() < 100) {
            trackingMap_ = MapManager::GetInstance().GetAllMappoints();
            cout << " Not enough active mappoints, reset tracking map to all mappoints" << endl;
        }
    }

    if (callback != nullptr) {
        callback();
    }

    cout << "Tracking map is updated" << endl;
}

void Frontend::MatchKeyPointsWithMappoints(const TrackingMap& trackingMap, size_t matchesToUseFlann)
{
    // Search for the matched keypoint for mappoints from local map
    matchedMptIdKptIdxMap_.clear();
    matchedKptIdxMptIdMap_.clear();
    matchedKptIdxDistanceMap_.clear();
    
    // mpt candidates that pass observation check for flann matching
    Mat flannMptCandidateDes;
    unordered_map<int, size_t> flannMptIdxToId; 

    // mpt candidates that not outlier, for flann matching
    Mat moreFlannMptCandidatesDes;
    unordered_map<int, size_t> moreFlannMptIdxToId; 

    size_t kptIdx;
    double distance;
    bool mayObserveMpt;
    bool hasMatchedKeypoint; 
    for (auto &[mptId, mpt] : trackingMap)
    {
        // If considered as outlier by backend or cannot be viewed by current frame
        // TODO: should remove this mappoint from the trackingMap
        if (mpt->outlier_) {
            continue;
        }

        hasMatchedKeypoint = frameCurr_->GetMatchedKeypoint(mpt, kptIdx, distance, mayObserveMpt);

        // Construct candidate for flann
        moreFlannMptIdxToId[moreFlannMptCandidatesDes.rows] = mptId;
        moreFlannMptCandidatesDes.push_back(mpt->GetDescriptor());
        if (mayObserveMpt) {
            flannMptIdxToId[flannMptCandidateDes.rows] = mptId;
            flannMptCandidateDes.push_back(mpt->GetDescriptor());
        }

        if (!hasMatchedKeypoint) {
            continue;
        }

        // Check whether this keypoint already has a better matched mappoint
        if (matchedKptIdxDistanceMap_.count(kptIdx)) {
            if (distance < matchedKptIdxDistanceMap_[kptIdx]) {
                // Remove the previous matched mappoint
                size_t mptIdToRemove = matchedKptIdxMptIdMap_[kptIdx];
                matchedMptIdKptIdxMap_.erase(mptIdToRemove);
            } else {
                continue;
            }
        }

        // add as a match
        matchedMptIdKptIdxMap_[mptId] = kptIdx;
        matchedKptIdxMptIdMap_[kptIdx] = mptId;
        matchedKptIdxDistanceMap_[kptIdx] = distance;
    }
    assert(matchedMptIdKptIdxMap_.size() == matchedKptIdxDistanceMap_.size());

    // If not found enough matches, fallback to use flann
    // if (matchedMptIdKptIdxMap_.size() < matchesToUseFlann) {
    //     cout << "  Fallback to use Flann matching" << endl;
    //     MatchKeyPointsFlann(flannMptCandidateDes, flannMptIdxToId);
    // }
    assert(matchedMptIdKptIdxMap_.size() == matchedKptIdxDistanceMap_.size());

    if (matchedMptIdKptIdxMap_.size() < matchesToUseFlann) {
        cout << "  Fallback to use all mappoints for Flann matching" << endl;
        MatchKeyPointsFlann(moreFlannMptCandidatesDes, moreFlannMptIdxToId);
    }
    assert(matchedMptIdKptIdxMap_.size() == matchedKptIdxDistanceMap_.size());

    cout << "  Size of tracking map: " << trackingMap.size() << endl;
    cout << "  Size of matched <mappoint, keypoint> pairs: " << matchedMptIdKptIdxMap_.size() << endl;
}

void Frontend::MatchKeyPointsFlann(const Mat& flannMptCandidateDes, unordered_map<int, size_t>& flannMptIdxToId) {
    matchedMptIdKptIdxMap_.clear();
    matchedKptIdxMptIdMap_.clear();
    matchedKptIdxDistanceMap_.clear();

    if (flannMptCandidateDes.rows == 0) {
        return;
    }

    vector<cv::DMatch> matches;
    flannMatcher_.match(flannMptCandidateDes, frameCurr_->GetDescriptors(), matches);

    // compute the min distance of the best match
    float min_dis = std::min_element(
                        matches.begin(),
                        matches.end(),
                        [](const cv::DMatch &m1, const cv::DMatch &m2)
                        { return m1.distance < m2.distance; })
                        ->distance;
    float maxDis = max<float>(min_dis * minDisRatio_, 30.0);

    for (cv::DMatch &m : matches)
    {
        // filter out the matches whose distance is large
        if (m.distance <= maxDis)
        {
            auto& mptId = flannMptIdxToId[m.queryIdx];
            auto& kptIdx = m.trainIdx;

            // Check whether this keypoint already has a better matched mappoint
            if (matchedKptIdxDistanceMap_.count(kptIdx)) {
                if (m.distance < matchedKptIdxDistanceMap_[kptIdx]) {
                    // Remove the previous matched mappoint
                    size_t mptIdToRemove = matchedKptIdxMptIdMap_[kptIdx];
                    matchedMptIdKptIdxMap_.erase(mptIdToRemove);
                } else {
                    continue;
                }
            }

            matchedMptIdKptIdxMap_[mptId] = kptIdx;
            matchedKptIdxMptIdMap_[kptIdx] = mptId;
            matchedKptIdxDistanceMap_[kptIdx] = m.distance;
        }
    }
}


void Frontend::EstimatePoseMotionOnlyBA(TrackingMap& trackingMap)
{
    // construct the 3d 2d observations
    vector<size_t> mptIds;
    vector<size_t> kptIdxs;
    vector<Point3f> pts3d;
    vector<Point2f> pts2d;

    for (auto& [mptId, kptIdx] : matchedMptIdKptIdxMap_) {
        mptIds.push_back(mptId);
        kptIdxs.push_back(kptIdx);
        pts3d.push_back(toPoint3f(trackingMap[mptId]->GetPosition()));
        pts2d.push_back(frameCurr_->GetKeypoint(kptIdx).pt);
    }

    // use P3P to compute the initial pose
    Mat initRotMat, rotVec, tranVec, inliers;
    cv::eigen2cv(frameCurr_->GetPose().rotationMatrix(), initRotMat);
    cv::Rodrigues(initRotMat, rotVec);
    cv::eigen2cv(frameCurr_->GetPose().translation(), tranVec);

    cv::solvePnPRansac(pts3d, pts2d, frameCurr_->camera_->GetCameraMatrix(), Mat(),
                        rotVec, tranVec, true,
                        100, 4.0, 0.99,
                        inliers, cv::SOLVEPNP_P3P);

    numInliers_ = inliers.rows;
    assert(numInliers_ != 0);
    cout << "  Size of inlier after P3P ransac: " << numInliers_ << endl;

    // Covert rotation vector to matrix and to eigen types
    Mat rotMat;
    cv::Rodrigues(rotVec, rotMat);
    Eigen::Matrix3d rotMatEigen;
    Vector3d tranVecEigen;
    cv::cv2eigen(rotMat, rotMatEigen);
    cv::cv2eigen(tranVec, tranVecEigen);
    SE3 pnpEstimatedPose = SE3(rotMatEigen, tranVecEigen);

    // clear previous allocated vertex and edge
    optimizer_.clear();

    VertexPose *poseVertex = new VertexPose();
    poseVertex->setId(0);
    poseVertex->setEstimate(pnpEstimatedPose);
    optimizer_.addVertex(poseVertex);

    // edges
    vector<UnaryEdgeProjection *> edges;

    for (size_t i = 0; i < inliers.rows; ++i)
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        UnaryEdgeProjection *edge = new UnaryEdgeProjection(toVec3d(pts3d[index]), frameCurr_->camera_);

        edge->setId(i);
        edge->setVertex(0, poseVertex);
        edge->setMeasurement(toVec2d(pts2d[index]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        auto rk = new g2o::RobustKernelHuber();
        rk->setDelta(sqrt(7.815));
        edge->setRobustKernel(rk);

        edges.push_back(edge);
        optimizer_.addEdge(edge);
    }

    // first round optimization
    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // remove edge outliers
    numInliers_ = 0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        auto& edge = edges[i];
        edge->computeError();

        // chi2 is the (u^2 + v^2)
        if (edge->chi2() > baInlierThres_) {
            // level 1 edges won't be optimized later
            edge->setLevel(1);
        }

        edge->setRobustKernel(nullptr);

        auto idx = inliers.at<int>(i, 0);
        ++numInliers_;
    }
    cout << "  Size of inlier after 1st BA: " << numInliers_ << endl;

    // Second round of BA. Reinitialize to only optimize level 0 edges
    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // Collect the inlier points
    baInlierMptIdSet_.clear();
    baInlierKptIdxSet_.clear();
    numInliers_ = 0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        auto& edge = edges[i];
        edge->computeError();

        if (edge->level() != 0 || edge->chi2() > baInlierThres_) {
            continue;
        }

        auto idx = inliers.at<int>(i, 0);
        baInlierMptIdSet_.insert(mptIds[idx]);
        baInlierKptIdxSet_.insert(kptIdxs[idx]);
        ++numInliers_;
    }
    cout << "  Size of inlier after 2nd BA " << numInliers_ << endl;

    // Set computed pose
    frameCurr_->SetPose(poseVertex->estimate());

    // TODO: remove the outliers from active map?
}

void Frontend::CreateTempMappoints() {
    lastFrameMpts_.clear();
    tempMpts_.clear();
    tempMptIdToKptIdx_.clear();
    for (size_t kptIdx = 0; kptIdx < frameCurr_->GetKeypointsSize(); ++kptIdx)
    {
        // temp mappoint doesn't have matched previous mappoint
        if (baInlierKptIdxSet_.count(kptIdx)) {
            auto& mpt = trackingMap_[matchedKptIdxMptIdMap_[kptIdx]];
            lastFrameMpts_[mpt->GetId()] = mpt;
            continue;
        }

        auto& kpt = frameCurr_->GetKeypoint(kptIdx);
        double depth = frameCurr_->GetDepth(kpt);
        if (depth < 0) {
            continue;
        }

        Vector3d mptPos = frameCurr_->camera_->Pixel2World(
            kpt, frameCurr_->GetPose(), depth);
        
        // create a mappoint
        // all parameters will have a deep copy inside the constructor
        Mappoint::Ptr mpt = Mappoint::CreateMappoint(mptPos);
        mpt->SetTempDescriptor(frameCurr_->GetDescriptor(kptIdx));

        lastFrameMpts_[mpt->GetId()] = mpt;
        tempMpts_.push_back(mpt);
        tempMptIdToKptIdx_[mpt->GetId()] = kptIdx;
    }
    cout << "Created temp mappoints: " << tempMpts_.size() << endl;
}

bool Frontend::IsGoodEstimation()
{
    // check if inliers number meet the threshold
    if (numInliers_ < minInliersForGood_)
    {
        cout << "Current tracking is rejected because inlier is too small: " << numInliers_ << endl;
        return false;
    }
    // check if the motion is too large
    SE3 T_r_c = framePrev_->GetPose() * frameCurr_->GetPose().inverse();
    Sophus::Vector6d d = T_r_c.log();
    if (d.norm() > 5.0)
    {
        cout << "Current tracking is rejected because motion is too large: " << d.norm() << endl;
        return false;
    }
    return true;
}

bool Frontend::IsKeyframe()
{
    if (numInliers_ < minInliersForKeyframe_) {
        return true;
    }

    SE3 T_r_c = framePrev_->GetPose() * frameCurr_->GetPose().inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if (rot.norm() > keyFrameMinRot_ || trans.norm() > keyFrameMinTrans_)
    {
        return true;
    }
    return false;
}

void Frontend::AddTempMappointsToMapManager()
{
    for (auto& mpt: tempMpts_) {
        MapManager::GetInstance().InsertMappoint(mpt);
    }
    cout << "Add temp mpts to map manager" << endl;
}

void Frontend::AddObservingMappointsToCurrentFrame() {
    // add matched previous mpts observation
    for (const auto& mptId : baInlierMptIdSet_) {
        const auto& kptIdx = matchedMptIdKptIdxMap_[mptId];
        frameCurr_->AddObservingMappoint(trackingMap_[mptId], kptIdx);
    }

    // add new created mpts observations
    for (const auto& mpt: tempMpts_) {
        auto& kptIdx = tempMptIdToKptIdx_[mpt->GetId()];
        frameCurr_->AddObservingMappoint(mpt, kptIdx);
    }
}

void Frontend::AddNewMappointsObservationsForOldKeyframes() {
    if (tempMpts_.size() == 0) {
        return;
    }

    auto localKeyframes = keyframeForTrackingMap_->GetCovisibleKeyframes();
    localKeyframes.insert(keyframeForTrackingMap_->GetId());

    for (auto& keyframeId : localKeyframes) {

        auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeId);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        // TODO: use previous keypoint as mask
        keyframe->ExtractKeyPointsAndComputeDescriptors(orb_);

        // Select the good mappoints candidates
        vector<Mappoint::Ptr> mptCandidates;
        Mat mptCandidatesDescriptors;
        for (auto & mappoint : tempMpts_)
        {
            // if ( !keyframe->IsCouldObserveMappoint(mappoint) ) {
            //     continue;
            // }

            // add as a candidate
            mptCandidates.push_back(mappoint);
            mptCandidatesDescriptors.push_back(mappoint->GetDescriptor());
        }

        vector<cv::DMatch> matches;
        flannMatcher_.match(mptCandidatesDescriptors, descriptors, matches);

        // compute the min distance of the best match
        float min_dis = std::min_element(
                            matches.begin(),
                            matches.end(),
                            [](const cv::DMatch &m1, const cv::DMatch &m2)
                            { return m1.distance < m2.distance; })
                            ->distance;

        int matchedSize = 0;
        for (cv::DMatch &m : matches)
        {
            // filter out the matches whose distance is large
            if (m.distance < max<float>(min_dis * minDisRatio_, 30.0))
            {
                ++matchedSize;
                // keyframe->AddObservingMappoint(mptCandidates[m.queryIdx]->GetId());
                // mptCandidates[m.queryIdx]->AddObservedByKeyframe(keyframe->GetId(), keypoints[m.trainIdx].pt);
            }
        }

        cout << " for keyframe " << keyframeId << " add " << matchedSize << " new observations \n";
    }
}

void Frontend::TriangulateMappointsInTrackingMap()
{
    int triangulatedCnt = 0;
    for (auto &[id, mpt] : trackingMap_)
    {
        if (mpt->outlier_ || mpt->triangulated_ || mpt->optimized_ || !baInlierMptIdSet_.count(id))
        {
            continue;
        }

        // try to triangulate the mappoint
        vector<SE3> poses;
        vector<Vec3> points;
        for (auto &[keyframeId, kptIdx] : mpt->GetObservedByKeyframesMap())
        {
            auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeId);
            auto& kpt = keyframe->GetKeypoint(kptIdx);

            if (keyframe == nullptr) {
                continue;
            }

            poses.push_back(keyframe->GetPose());
            points.push_back(keyframe->camera_->Pixel2Camera(kpt.pt));
        }

        if (poses.size() >= 2)
        {
            Vec3 pworld = Vec3::Zero();
            if (Triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                // if triangulate successfully
                mpt->SetPosition(pworld);
                mpt->triangulated_ = true;
                triangulatedCnt++;
                break;
            }
        }
    }
    cout << "  Triangulate active mappoints size: " << triangulatedCnt << endl;
}

} // namespace