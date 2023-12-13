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

FrontEnd::FrontEnd()    
{
    state_ = INITIALIZING;

    flannMatcher_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 10, 2));

    orb_ = cv::ORB::create(Config::get<int>("number_of_features"),
                            Config::get<double>("scale_factor"),
                            Config::get<int>("level_pyramid"));
    minDisRatio_ = Config::get<float>("match_ratio");
    maxLostFrames_ = Config::get<float>("max_num_lost");
    minInliers_ = Config::get<int>("min_inliers");
    keyFrameMinRot_ = Config::get<double>("keyframe_rotation");
    keyFrameMinTrans_ = Config::get<double>("keyframe_translation");
}

bool FrontEnd::AddFrame(const Frame::Ptr frame)
{
    cout << "Frontend status: " << VOStateStr[state_] << endl;
    frameCurr_ = frame;

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
        viewer_->setCurrentFrame(frameCurr_, matchedKptSet_);
        viewer_->updateDrawingObjects();
    }

    return true;
}

void FrontEnd::InitializationHandler() {
    ExtractKeyPointsAndComputeDescriptors();

    // the first frame is a keyframe
    MapManager::GetInstance().InsertKeyframe(frameCurr_);
    CreateNewMappoints();

    // RGBD camera only needs 1 frame to configure since it could get the depth information
    state_ = TRACKING;
    framePrev_ = frameCurr_;
    keyframeRef_ = frameCurr_;
}

bool FrontEnd::TrackingHandler() {
    // set an initial pose to the pose of previous pose, used for filtering mappoints candidate in tracking map
    frameCurr_->SetPose(framePrev_->GetPose());

    ExtractKeyPointsAndComputeDescriptors();

    // Coarse compute pose
    cout << "Corase computing...\n";
    MatchKeyPointsInTrackingMap();
    EstimatePosePnP();

    // Since the pose of frame is updated, try again to get more mappint candidates and more matches
    cout << "Fine computing...\n";
    MatchKeyPointsInTrackingMap();
    EstimatePosePnP();

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

    AddMatchedMappointsToKeyframeObservations();
    CreateNewMappoints();
    AddNewMappointsObservationsForOldKeyframes();
    
    TriangulateMappointsInTrackingMap();

    // if have backend, use backend to optimize mappoints position and frame pose
    if (backend_)
    {
        backend_->OptimizeCovisibleGraphOfKeyframe(frameCurr_);
    }

    framePrev_ = frameCurr_;
    keyframeRef_ = frameCurr_;

    return true;
}

void FrontEnd::LostHandler() {
    cout << "Tracking is lost" << endl;
}

void FrontEnd::ExtractKeyPointsAndComputeDescriptors()
{
    keypointsCurr_.clear();
    orb_->detectAndCompute(frameCurr_->color_, Mat(), keypointsCurr_, descriptorsCurr_);
}

void FrontEnd::MatchKeyPointsInTrackingMap()
{
    // Update the tracking map
    if (keyframeForTrackingMap_ != keyframeRef_) {
        keyframeForTrackingMap_ = keyframeRef_;
        trackingMap_ = MapManager::GetInstance().GetMappointsAroundKeyframe(keyframeRef_);
    }
    if (trackingMap_.size() < 100) {
        trackingMap_ = MapManager::GetInstance().GetAllMappoints();
        cout << " Not enough active mappoints, reset tracking map to all mappoints" << endl;
    }

    // Select the good mappoints candidates
    vector<Mappoint::Ptr> mptCandidates;
    Mat mptCandidatesDescriptors;
    for (auto &mappointIdToPtr : trackingMap_)
    {
        auto mp = mappointIdToPtr.second;

        // If considered as outlier by backend or cannot be viewed by current frame
        // TODO: should remove this mappoint from the trackingMap_
        if ( mp->outlier_ || !frameCurr_->IsCouldObserveMappoint(mp) ) {
            continue;
        }

        // add as a candidate
        mptCandidates.push_back(mp);
        mptCandidatesDescriptors.push_back(mp->descriptor_);
    }

    vector<cv::DMatch> matches;
    flannMatcher_.match(mptCandidatesDescriptors, descriptorsCurr_, matches);

    // compute the min distance of the best match
    float min_dis = std::min_element(
                        matches.begin(),
                        matches.end(),
                        [](const cv::DMatch &m1, const cv::DMatch &m2)
                        { return m1.distance < m2.distance; })
                        ->distance;

    cout << "Minimum distance of matches " << min_dis << endl;
    cout << "Largest distance of matches " << max<float>(min_dis * minDisRatio_, 30.0) << endl;

    flannMatchedMptKptMap_.clear();
    matchedKptSet_.clear();
    for (cv::DMatch &m : matches)
    {
        // filter out the matches whose distance is large
        if (m.distance < max<float>(min_dis * minDisRatio_, 30.0))
        {
            flannMatchedMptKptMap_[mptCandidates[m.queryIdx]] = keypointsCurr_[m.trainIdx];
            matchedKptSet_.insert(keypointsCurr_[m.trainIdx]);
        }
    }
    cout << "  Size of tracking map: " << trackingMap_.size() << endl;
    cout << "  Size of mappoint candidates: " << mptCandidates.size() << endl;
    cout << "  Size of matched <mappoint, keypoint> pairs: " << flannMatchedMptKptMap_.size() << endl;
}

void FrontEnd::EstimatePosePnP()
{
    // construct the 3d 2d observations
    vector<Mappoint::Ptr> mpts3d;
    vector<Point3f> pts3d;
    vector<Point2f> pts2d;

    for (auto &mappointToKeypoint : flannMatchedMptKptMap_) {
        mpts3d.push_back(mappointToKeypoint.first);
        pts3d.push_back(toPoint3f(mappointToKeypoint.first->GetPosition()));
        pts2d.push_back(mappointToKeypoint.second.pt);
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
    cout << "  PNP results inlier size: " << numInliers_ << endl;

    // Covert rotation vector to matrix
    Mat rotMat;
    cv::Rodrigues(rotVec, rotMat);
    Eigen::Matrix3d rotMat_eigen;
    rotMat_eigen << rotMat.at<double>(0, 0), rotMat.at<double>(0, 1), rotMat.at<double>(0, 2),
        rotMat.at<double>(1, 0), rotMat.at<double>(1, 1), rotMat.at<double>(1, 2),
        rotMat.at<double>(2, 0), rotMat.at<double>(2, 1), rotMat.at<double>(2, 2);
    SE3 pnpEstimatedPose = SE3(
        rotMat_eigen,
        Vector3d(tranVec.at<double>(0, 0), tranVec.at<double>(1, 0), tranVec.at<double>(2, 0)));

    // using motion-only bundle adjustment to optimize the pose
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexPose *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(pnpEstimatedPose);
    optimizer.addVertex(pose);

    // edges
    vector<UnaryEdgeProjection *> edges;

    for (size_t i = 0; i < inliers.rows; ++i)
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        UnaryEdgeProjection *edge = new UnaryEdgeProjection(toVec3d(pts3d[index]), frameCurr_->camera_);

        edge->setId(i);
        edge->setVertex(0, pose);
        edge->setMeasurement(toVec2d(pts2d[index]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        auto rk = new g2o::RobustKernelHuber();
        rk->setDelta(sqrt(7.815));
        edge->setRobustKernel(rk);

        edges.push_back(edge);
        optimizer.addEdge(edge);
    }

    // first round optimization
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // remove edge outliers
    int outlierCnt = 0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        auto edge = edges[i];
        edge->computeError();

        if (edge->chi2() > 1) {
            edge->setLevel(1);
            outlierCnt++;
        }

        edge->setRobustKernel(0);
    }
    cout << "got outliders after first round BA: " << outlierCnt << endl;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    frameCurr_->SetPose(pose->estimate());

    pnpMatchedMptSet_.clear();
    for (size_t i = 0; i < edges.size(); ++i)
    {
        auto edge = edges[i];
        edge->computeError();

        if (edge->chi2() > 1) {
            continue;
        }

        auto mpt = mpts3d[inliers.at<int>(i, 0)];
        pnpMatchedMptSet_.insert(mpt);
    }

    // TODO: remove the outliers from active map?
}

bool FrontEnd::IsGoodEstimation()
{
    // check if inliers number meet the threshold
    if (numInliers_ < minInliers_)
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

bool FrontEnd::IsKeyframe()
{
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

void FrontEnd::AddMatchedMappointsToKeyframeObservations() {
    for (auto& mappoint : pnpMatchedMptSet_) {
        frameCurr_->AddObservedMappoint(mappoint->GetId(), flannMatchedMptKptMap_[mappoint].pt);
    }
}

void FrontEnd::CreateNewMappoints()
{
    newMappoints_.clear();
    for (size_t idx = 0; idx < keypointsCurr_.size(); ++idx)
    {
        // the new mappoint is keypoint doesn't match with previous mappoints
        if (matchedKptSet_.count(keypointsCurr_[idx])) {
            continue;
        }

        double depth = frameCurr_->GetDepth(keypointsCurr_[idx]);
        if (depth < 0) {
            continue;
        }

        Vector3d mptPos = frameCurr_->camera_->Pixel2World(
            keypointsCurr_[idx], frameCurr_->GetPose(), depth);

        // create a mappoint
        // all parameters will have a deep copy inside the constructor
        Mappoint::Ptr mpt = Mappoint::CreateMappoint(
            mptPos,
            descriptorsCurr_.row(idx));

        // add mappoint into map
        MapManager::GetInstance().InsertMappoint(mpt);

        // add new mappoint to keyframe observation
        frameCurr_->AddObservedMappoint(mpt->GetId(), keypointsCurr_[idx].pt);

        // record new mappoints
        newMappoints_.push_back(mpt);
    }
    cout << "Created new mappoints: " << newMappoints_.size() << endl;
}

void FrontEnd::AddNewMappointsObservationsForOldKeyframes() {
    if (newMappoints_.size() == 0) {
        return;
    }

    auto localKeyframes = keyframeRef_->GetCovisibleKeyframes();
    localKeyframes.insert(keyframeRef_->GetId());

    for (auto& keyframeId : localKeyframes) {

        auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeId);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        // TODO: use previous keypoint as mask
        orb_->detectAndCompute(keyframe->color_, Mat(), keypoints, descriptors);

        // Select the good mappoints candidates
        vector<Mappoint::Ptr> mptCandidates;
        Mat mptCandidatesDescriptors;
        for (auto & mappoint : newMappoints_)
        {
            if ( !keyframe->IsCouldObserveMappoint(mappoint) ) {
                continue;
            }

            // add as a candidate
            mptCandidates.push_back(mappoint);
            mptCandidatesDescriptors.push_back(mappoint->descriptor_);
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
                // keyframe->AddObservedMappoint(mptCandidates[m.queryIdx]->GetId());
                // mptCandidates[m.queryIdx]->AddObservedByKeyframe(keyframe->GetId(), keypoints[m.trainIdx].pt);
            }
        }

        cout << " for keyframe " << keyframeId << " add " << matchedSize << " new observations \n";
    }
}

void FrontEnd::TriangulateMappointsInTrackingMap()
{
    int triangulatedCnt = 0;
    for (auto &idToMappoint : trackingMap_)
    {
        auto mp = idToMappoint.second;
        if (mp->outlier_ || mp->triangulated_ || mp->optimized_ || !flannMatchedMptKptMap_.count(mp))
        {
            continue;
        }

        // try to triangulate the mappoint
        vector<SE3> poses;
        vector<Vec3> points;
        for (auto &keyframeIdToPixelPos : mp->GetObservedByKeyframesMap())
        {
            auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeIdToPixelPos.first);
            auto pixelPos = keyframeIdToPixelPos.second;

            if (keyframe == nullptr) {
                continue;
            }

            poses.push_back(keyframe->GetPose());
            points.push_back(keyframe->camera_->Pixel2Camera(pixelPos));
        }

        if (poses.size() >= 2)
        {
            Vec3 pworld = Vec3::Zero();
            if (Triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                // if triangulate successfully
                mp->SetPosition(pworld);
                mp->triangulated_ = true;
                triangulatedCnt++;
                break;
            }
        }
    }
    cout << "  Triangulate active mappoints size: " << triangulatedCnt << endl;
}

} // namespace