/*
 * 
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/util.h"

namespace myslam
{

    FrontEnd::FrontEnd() : state_(INITIALIZING), frameRef_(nullptr), frameCurr_(nullptr), accuLostFrameNums_(0), num_inliers_(0),
                           flannMatcher_(new cv::flann::LshIndexParams(5, 10, 2))
    {
        orb_ = cv::ORB::create(Config::get<int>("number_of_features"),
                               Config::get<double>("scale_factor"),
                               Config::get<int>("level_pyramid"));
        minDisRatio_ = Config::get<float>("match_ratio");
        maxLostFrames_ = Config::get<float>("max_num_lost");
        min_inliers_ = Config::get<int>("min_inliers");
        keyFrameMinRot_ = Config::get<double>("keyframe_rotation");
        keyFrameMinTrans_ = Config::get<double>("keyframe_translation");

        cout << "Frontend status: -1: Initialization, 0: Tracking, 1: Lost" << endl;
    }

    bool FrontEnd::addFrame(Frame::Ptr frame)
    {
        cout << "Frontend status: " << state_ << endl;
        frameCurr_ = frame;

        switch (state_)
        {
        case INITIALIZING:
        {
            // RGBD camera only needs 1 frame to configure since it could get the depth information
            state_ = TRACKING;

            // extract features from frameCurr_
            extractKeyPointsAndComputeDescriptors();

            // the first frame is a key-frame
            map_->insertKeyFrame(frameCurr_);
            initMap();
            frameRef_ = frame;
            break;
        }
        case TRACKING:
        {
            // set an initial pose, used for looking for map points in current view
            frameCurr_->setPose(frameRef_->getPose());

            extractKeyPointsAndComputeDescriptors();
            matchKeyPointsWithActiveMapPoints();
            estimatePosePnP();

            // bad estimation due to various reasons
            if (!isGoodEstimation())
            {
                cout << "Cannot estimate Pose" << endl;
                accuLostFrameNums_++;
                state_ = (++accuLostFrameNums_ > maxLostFrames_) ? LOST : TRACKING;
                return false;
            }

            // if good estimation, reset the num of lost
            accuLostFrameNums_ = 0;

            // set estimated pose to current frame
            frameCurr_->setPose(estimatedPoseCurr_);

            // remove non-active mappoints
            cullNonActiveMapPoints();

            if ( isKeyFrame() )
            {
                cout << "  Current frame is a new keyframe" << endl;
                map_->insertKeyFrame(frameCurr_);

                // Add this keyframe as the observation of observed mappoints
                addKeyframeObservationToOldMapPoints();

                // Add some new mappoints since current frame is keyframe
                addNewMapPoints();

                // Optimize active mappoints since has a new keyframe
                triangulateActiveMapPoints();
                
                // if have backend, use backend to optimize mappoints position and frame pose
                if (backend_) {
                    frameCurr_->updateConnectedKeyFrames();
                    backend_->optimizeCovisibilityGraph(frameCurr_);
                }

                frameRef_ = frameCurr_;
            }
            break;
        }
        case LOST:
        {
            cout << "Tracking has lost" << endl;
            break;
        }
        }

        viewer_->setCurrentFrame(frameCurr_, matchedKptSet_);
        viewer_->updateDrawingObjects();

        return true;
    }

    void FrontEnd::extractKeyPointsAndComputeDescriptors()
    {
        orb_->detectAndCompute(frameCurr_->color_, Mat(), keypointsCurr_, descriptorsCurr_);
    }

    void FrontEnd::matchKeyPointsWithActiveMapPoints()
    {
        // get the active mappoints candidates from map
        auto activeMpts = map_->getActiveMappoints();
        if (activeMpts.size() < 100)
        {
            map_->resetActiveMappoints();
            activeMpts = map_->getAllMappoints();
            cout << " Not enough active mappoints, reset activie mappoints to all mappoints" << endl;
        }

        // Select the good mappoints candidates
        vector<MapPoint::Ptr> mptCandidates;
        Mat descriptorCandidates;
        for (auto &mappoint : activeMpts)
        {
            auto mp = mappoint.second;

            // If considered as outlier by backend
            if (mp->outlier_)
            {
                continue;
            }

            if (frameCurr_->isInFrame(mp->getPosition()))
            {
                // add to candidate
                mp->visibleTimes_++;
                mptCandidates.push_back(mp);
                descriptorCandidates.push_back(mp->descriptor_);
            }
        }

        vector<cv::DMatch> matches;
        flannMatcher_.match(descriptorCandidates, descriptorsCurr_, matches);
        // select the best matches
        float min_dis = std::min_element(
                            matches.begin(),
                            matches.end(),
                            [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; })
                            ->distance;

        matchedMptKptMap_.clear();
        matchedKptSet_.clear();
        for (cv::DMatch &m : matches)
        {
            if (m.distance < max<float>(min_dis * minDisRatio_, 30.0))
            {
                matchedMptKptMap_[mptCandidates[m.queryIdx]] = keypointsCurr_[m.trainIdx];
                matchedKptSet_.insert(keypointsCurr_[m.trainIdx]);
            }
        }
        cout << "  Active mappoints size: " << mptCandidates.size() << endl;
        cout << "  Matched feature paris size: " << matchedMptKptMap_.size() << endl;
    }

    void FrontEnd::estimatePosePnP()
    {
        // construct the 3d 2d observations
        vector<MapPoint::Ptr> mpts3d;
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (auto &pair : matchedMptKptMap_)
        {
            mpts3d.push_back(pair.first);
            pts3d.push_back(toPoint3f(pair.first->getPosition()));
            pts2d.push_back(pair.second.pt);
        }

        // use P3P wih Ransac to solve an intial value of the pose
        Mat K = frameCurr_->camera_->getCameraMatrix();
        Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_P3P);

        num_inliers_ = inliers.rows;
        cout << "  PNP results inlier size: " << num_inliers_ << endl;

        // Covert rotation vector to matrix
        Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d R_eigen;
        R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        estimatedPoseCurr_ = SE3(
            R_eigen,
            Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

        // using motion-only bundle adjustment to optimize the pose
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        VertexPose *pose = new VertexPose();
        pose->setId(0);
        pose->setEstimate(estimatedPoseCurr_);
        optimizer.addVertex(pose);

        // edges
        for (int i = 0; i < inliers.rows; i++)
        {
            int index = inliers.at<int>(i, 0);
            // 3D -> 2D projection
            UnaryEdgeProjection *edge = new UnaryEdgeProjection(toVec3d(pts3d[index]), frameCurr_->camera_);
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->setMeasurement(toVec2d(pts2d[index]));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            // increase the inlier mappoints matched times
            mpts3d[index]->matchedTimes_++;
            // set this mappoint as the observed mappoints of current frame
            frameCurr_->addObservedMapPoint(mpts3d[index]);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        estimatedPoseCurr_ = pose->estimate();
    }

    bool FrontEnd::isGoodEstimation()
    {
        // check if inliers number meet the threshold
        if (num_inliers_ < min_inliers_)
        {
            cout << "Current tracking is rejected because inlier is too small: " << num_inliers_ << endl;
            return false;
        }
        // check if the motion is too large
        SE3 T_r_c = frameRef_->getPose() * estimatedPoseCurr_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        if (d.norm() > 5.0)
        {
            cout << "Current tracking is rejected because motion is too large: " << d.norm() << endl;
            return false;
        }
        return true;
    }

    bool FrontEnd::isKeyFrame()
    {
        SE3 T_r_c = frameRef_->getPose() * estimatedPoseCurr_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > keyFrameMinRot_ || trans.norm() > keyFrameMinTrans_) {
            return true;
        }
        return false;
    }

    void FrontEnd::initMap()
    {
        for (size_t i = 0; i < keypointsCurr_.size(); i++)
        {
            addNewMapPoint(i);
        }
    }

    void FrontEnd::cullNonActiveMapPoints()
    {
        map_->cullNonActiveMapPoints(frameCurr_);
        map_->updateMappointEraseRatio();

        cout << "  Active mappoints size after culling: " << map_->getActiveMappoints().size() << endl;
    }

    void FrontEnd::addNewMapPoints()
    {
        for (size_t i = 0; i < keypointsCurr_.size(); i++)
        {
            // if the keypoint doesn't match with previous mappoints, this is a new mappoint
            if ( !matchedKptSet_.count(keypointsCurr_[i]) )
            {
                addNewMapPoint(i);
            }
        }
        map_->updateMappointEraseRatio();
        cout << "  Active mappoints size after adding: " << map_->getActiveMappoints().size() << endl;
    }

    void FrontEnd::addNewMapPoint(const int &idx)
    {

        double depth = frameCurr_->findDepth(keypointsCurr_[idx]);
        if (depth < 0) {
            return;
        }

        Vector3d mptPos = frameCurr_->camera_->pixel2world(
            keypointsCurr_[idx], frameCurr_->getPose(), depth);

        // Create a mappoint
        // deep copy the descriptor and keypoint position since they will be clear in next track
        MapPoint::Ptr mpt = MapPoint::createMapPoint(
            mptPos,
            (mptPos - frameCurr_->getCamCenter()).normalized(),
            descriptorsCurr_.row(idx).clone(),
            frameCurr_,
            cv::Point2f(keypointsCurr_[idx].pt));

        // set this mappoint as the observed mappoints of current frame
        frameCurr_->addObservedMapPoint(mpt);

        // Add mappoint into map
        map_->insertMapPoint(mpt);
    }

    void FrontEnd::addKeyframeObservationToOldMapPoints() {
        for (auto& mappoint : frameCurr_ -> getObservedMapPoints()) {
            if (mappoint.expired()) {
                continue;
            }

            auto mp = mappoint.lock();
            if (mp -> outlier_) {
                continue;
            }

            mp -> addKeyFrameObservation(frameCurr_, cv::Point2f(matchedMptKptMap_[mp].pt));
        }
    }

    void FrontEnd::triangulateActiveMapPoints()
    {
        int triangulatedCnt = 0;
        for (auto &mappoint : map_->getActiveMappoints())
        {
            auto mp = mappoint.second;
            if ( mp->outlier_ || mp->triangulated_) {
                continue;
            }

            // it current keyframe doesn't observe this mappoint, it cannot be triangulated
            if ( !matchedMptKptMap_.count(mp) ) {
                continue;
            }

            // try to triangulate the mappoint
            vector<SE3> poses;
            vector<Vec3> points;
            for (auto &keyFrameMap : mp->getKeyFrameObservationsMap())
            {
                if (keyFrameMap.first.expired()) {
                    continue;
                }

                auto keyFrame = keyFrameMap.first.lock();
                auto keyPoint = keyFrameMap.second;

                poses.push_back(keyFrame->getPose());
                points.push_back(keyFrame->camera_->pixel2camera(keyPoint));
            }

            if (poses.size() >= 2) {
                Vec3 pworld = Vec3::Zero();
                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    // if triangulate successfully
                    mp->setPosition(pworld);
                    mp->triangulated_ = true;
                    triangulatedCnt++;
                    break;
                }
            }
        }
        cout << "  Triangulate active mappoints size: " << triangulatedCnt << endl;
    }

} //namespace