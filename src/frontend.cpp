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

Frontend::Frontend(const Camera::Ptr& camera) {

    camera_ = camera;

    flannMatcher_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 10, 2));

    // Read paras
    orb_ = cv::ORB::create(Config::get<int>("number_of_features") / (Config::get<int>("row_section_cnt") * Config::get<int>("col_section_cnt")),
                            Config::get<double>("scale_factor"),
                            Config::get<int>("level_pyramid"));
    useActiveSearch_ = myslam::Config::get<int> ("use_feature_active_search");
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

    // Setup backend
    backend_ = Backend::Ptr(new myslam::Backend(camera_));
    backend_->RegisterTrackingMapUpdateCallback(
            [&](function<void(TrackingMap&)> updater) {
                UpdateTrackingMap(updater);
            });

    state_ = INITIALIZING;
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
    MapManager::Instance().AddKeyframe(frameCurr_);
    CreateTempMappoints();
    assert(lastFrameMpts_.size() == tempMptKptIdxMap_.size());
    for(auto& [mptId, mpt]: lastFrameMpts_) {
        MapManager::Instance().AddMappoint(mpt);
        frameCurr_->AddObservingMappoint(mpt, tempMptKptIdxMap_[mpt]);
    }
    UpdateTrackingMap([&](TrackingMap& trackingMap) {
        trackingMap.clear();
        trackingMap.insert(lastFrameMpts_.begin(), lastFrameMpts_.end());
    });

    // RGBD camera only needs 1 frame to configure since it could get the depth information
    state_ = TRACKING;
}

bool Frontend::TrackingHandler() {
    // lock tracking map, so it cannot be updated during frontend processing
    unique_lock<mutex> lock(trackingMapMutex_);

    // set an initial pose to the pose of previous pose, used for feature matching
    frameCurr_->SetTcw(framePrev_->GetTcw());

    // Compute pose based on last frame mappoints
    cout << "Frame tracking...\n";
    MatchKeyPointsWithMappoints(lastFrameMpts_, false, minMatchesToUseFlannFrameTracking_);
    EstimatePoseMotionOnlyBA(lastFrameMpts_);

    // Compute pose based on tracking map
    cout << "Map tracking...\n";
    MatchKeyPointsWithMappoints(trackingMap_, true, minMatchesToUseFlannMapTracking_);
    EstimatePoseMotionOnlyBA(trackingMap_);

    // Create temp mappoints for next frame tracking
    CreateTempMappoints();

    if (!IsGoodEstimation()) {
        cout << "Cannot estimate Pose" << endl;
        accuLostFrameNums_++;
        state_ = (++accuLostFrameNums_ > maxLostFrames_) ? LOST : TRACKING;
        return false;
    } 
    accuLostFrameNums_ = 0;
    
    if (!IsKeyframe()) {
        return true;
    } 
    cout << "Current frame is a new keyframe" << endl;
    
    backend_->AddNewKeyframeInfo({frameCurr_, baInlierMptIdKptIdxMap_, tempMptKptIdxMap_});

    return true;
}

void Frontend::LostHandler() {
    cout << "Tracking is lost" << endl;
}

void Frontend::UpdateTrackingMap(function<void(TrackingMap&)> updater) {
    unique_lock<mutex> lock(trackingMapMutex_);

    // Use updater to update tracking map
    updater(trackingMap_);

    cout << "Tracking map is updated" << endl;
}

void Frontend::MatchKeyPointsWithMappoints(const TrackingMap& trackingMap, const bool doDirectionCheck, const size_t matchesToUseFlann)
{
    // Search for the matched keypoint for mappoints from local map
    matchedMptIdKptIdxMap_.clear();
    matchedKptIdxMptIdMap_.clear();
    matchedKptIdxDistanceMap_.clear();
    
    // mpt candidates that pass observation check for flann matching
    // Mat flannMptCandidateDes;
    // unordered_map<int, size_t> flannMptIdxToId; 

    // mpt candidates that not outlier, for flann matching
    Mat moreFlannMptCandidatesDes;
    unordered_map<int, size_t> moreFlannMptIdxToId; 

    size_t kptIdx;
    double distance;
    bool mayObserveMpt;
    bool hasMatchedKeypoint; 
    for (auto &[mptId, mpt] : trackingMap)
    {
        // mpt in tracking map are guranteed to be non-outlier and has been optimized by backend. But check in case
        if (mpt->outlier_) {
            continue;
        }

        // Construct candidate for flann
        moreFlannMptIdxToId[moreFlannMptCandidatesDes.rows] = mptId;
        moreFlannMptCandidatesDes.push_back(mpt->GetDescriptor());

        // If not using active search, just continue here to avoid extra search
        if (!useActiveSearch_) {
            continue;
        }

        hasMatchedKeypoint = frameCurr_->GetMatchedKeypoint(mpt, doDirectionCheck, kptIdx, distance, mayObserveMpt);

        // if (mayObserveMpt) {
        //     flannMptIdxToId[flannMptCandidateDes.rows] = mptId;
        //     flannMptCandidateDes.push_back(mpt->GetDescriptor());
        // }

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

    if (!useActiveSearch_ || matchedMptIdKptIdxMap_.size() < matchesToUseFlann) {
        MatchKeyPointsFlann(moreFlannMptCandidatesDes, moreFlannMptIdxToId);

        if (flannMatchedMptIdKptIdxMap_.size() > matchedMptIdKptIdxMap_.size()) {
            if (useActiveSearch_) {
                printf("  Active searched matched size: %zu is too mall, fallback to use Flann matching\n", matchedMptIdKptIdxMap_.size());
            }

            matchedMptIdKptIdxMap_.clear();
            matchedMptIdKptIdxMap_.insert(flannMatchedMptIdKptIdxMap_.begin(), flannMatchedMptIdKptIdxMap_.end());
        
            matchedKptIdxMptIdMap_.clear();
            matchedKptIdxMptIdMap_.insert(flannMatchedKptIdxMptIdMap_.begin(), flannMatchedKptIdxMptIdMap_.end());

            matchedKptIdxDistanceMap_.clear();
            matchedKptIdxDistanceMap_.insert(flannMatchedKptIdxDistanceMap_.begin(), flannMatchedKptIdxDistanceMap_.end());
        }
    }
    assert(matchedMptIdKptIdxMap_.size() == matchedKptIdxDistanceMap_.size());

    cout << "  Size of tracking map: " << trackingMap.size() << endl;
    cout << "  Size of matched <mappoint, keypoint> pairs: " << matchedMptIdKptIdxMap_.size() << endl;
}

void Frontend::MatchKeyPointsFlann(const Mat& flannMptCandidateDes, unordered_map<int, size_t>& flannMptIdxToId) {
    flannMatchedMptIdKptIdxMap_.clear();
    flannMatchedKptIdxMptIdMap_.clear();
    flannMatchedKptIdxDistanceMap_.clear();

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
            if (flannMatchedKptIdxDistanceMap_.count(kptIdx)) {
                if (m.distance < flannMatchedKptIdxDistanceMap_[kptIdx]) {
                    // Remove the previous matched mappoint
                    size_t mptIdToRemove = flannMatchedKptIdxMptIdMap_[kptIdx];
                    flannMatchedMptIdKptIdxMap_.erase(mptIdToRemove);
                } else {
                    continue;
                }
            }

            flannMatchedMptIdKptIdxMap_[mptId] = kptIdx;
            flannMatchedKptIdxMptIdMap_[kptIdx] = mptId;
            flannMatchedKptIdxDistanceMap_[kptIdx] = m.distance;
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
    cv::eigen2cv(frameCurr_->GetTcw().rotationMatrix(), initRotMat);
    cv::Rodrigues(initRotMat, rotVec);
    cv::eigen2cv(frameCurr_->GetTcw().translation(), tranVec);

    cv::solvePnPRansac(pts3d, pts2d, camera_->GetCameraMatrix(), Mat(),
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

    for (size_t i = 1; i < inliers.rows; ++i)
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        UnaryEdgeProjection *edge = new UnaryEdgeProjection(toVector3d(pts3d[index]), camera_);
        edge->setId(i);
        edge->setVertex(0, poseVertex);
        edge->setMeasurement(toVec2d(pts2d[index]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        auto rk = new g2o::RobustKernelHuber();
        rk->setDelta(sqrt(baInlierThres_));
        edge->setRobustKernel(rk);

        edges.push_back(edge);
        optimizer_.addEdge(edge);
    }

    vector<bool> edgeIsOutlier (edges.size(), false);
    for (size_t iteration = 0; iteration < 4; ++iteration) {
        poseVertex->setEstimate(pnpEstimatedPose);
        optimizer_.initializeOptimization(0);
        optimizer_.optimize(10);

        // Handle outliers
        for (size_t i = 0; i < edges.size(); ++i)
        {
            auto& edge = edges[i];
            if (edgeIsOutlier[i]) {
                edge->computeError();
            }

            // chi2 is the (u^2 + v^2)
            if (edge->chi2() > baInlierThres_) {
                // level 1 edges won't be optimized later
                edge->setLevel(1);
                edgeIsOutlier[i] = true;
            } else {
                edge->setLevel(0);
                edgeIsOutlier[i] = false;
            }

            if (iteration == 2) {
                edge->setRobustKernel(nullptr);
            }
        }
    }

    // Collect the inlier points
    baInlierMptIdKptIdxMap_.clear();
    baInlierKptIdxSet_.clear();
    numInliers_ = 0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        if (edgeIsOutlier[i]) {
            continue;
        }

        auto idx = inliers.at<int>(i, 0);
        auto& mptId = mptIds[idx];
        auto& kptIdx = kptIdxs[idx];
        baInlierMptIdKptIdxMap_[mptId] = kptIdx;
        baInlierKptIdxSet_.insert(kptIdx);
        ++numInliers_;
    }
    cout << "  Size of inlier after BA " << numInliers_ << endl;

    // Set computed pose
    frameCurr_->SetTcw(poseVertex->estimate());

    // TODO: remove the outliers from active map?
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
    SE3 T_r_c = framePrev_->GetTcw() * frameCurr_->GetTcw().inverse();
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

    SE3 T_r_c = framePrev_->GetTcw() * frameCurr_->GetTcw().inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if (rot.norm() > keyFrameMinRot_ || trans.norm() > keyFrameMinTrans_)
    {
        return true;
    }
    return false;
}

void Frontend::CreateTempMappoints() {
    lastFrameMpts_.clear();
    tempMptKptIdxMap_.clear();
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

        Vector3d mptPos = camera_->Pixel2World(
            kpt, frameCurr_->GetTcw(), depth);
        
        // create a mappoint
        // all parameters will have a deep copy inside the constructor
        Mappoint::Ptr mpt = Mappoint::CreateMappoint(mptPos, frameCurr_->GetDescriptor(kptIdx));

        lastFrameMpts_[mpt->GetId()] = mpt;
        tempMptKptIdxMap_[mpt] = kptIdx;
    }
    cout << "Created temp mappoints: " << tempMptKptIdxMap_.size() << endl;
}

void Frontend::Stop() {
    backend_->Stop();
    if (viewer_ != nullptr) {
        viewer_->Stop();
    }
}

} // namespace