/*
 * Fontend which tracks camera poses on frames
 *
 * The entry point is the AddFrame function, which gets a frame pointer and computes the camera pose on that frame. Return false if the tracking fails.
 *
 * Frontend is also reponsible for the other functions
 * 1. create keyframe
 * 2. invoke backend (if there is) to optimize
 * 3. invoke reviewer (if there is) to show the image frames, real-time poses and maps
 */

#include "myslam/frontend.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/g2o_types.h"
#include "myslam/util.h"
#include "myslam/mapmanager.h"

namespace myslam
{

    FrontEnd::FrontEnd() : state_(INITIALIZING), framePrev_(nullptr), frameCurr_(nullptr), accuLostFrameNums_(0), num_inliers_(0),
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
            // extract features from frameCurr_
            extractKeyPointsAndComputeDescriptors();

            // the first frame is a key-frame
            MapManager::GetInstance().InsertKeyframe(frameCurr_);
            for (size_t i = 0; i < keypointsCurr_.size(); i++)
            {
                addNewMapPoint(i);
            }

            // RGBD camera only needs 1 frame to configure since it could get the depth information
            state_ = TRACKING;
            framePrev_ = frame;
            keyframeRef_ = frame;
            break;
        }
        case TRACKING:
        {
            // set an initial pose to the pose of previous pose, used for looking for map points in current view
            frameCurr_->setPose(framePrev_->getPose());

            extractKeyPointsAndComputeDescriptors();

            // Corase compute pose
            cout << "Corase computing...\n";
            matchKeyPointsWithActiveMapPoints();
            estimatePosePnP(false);
            frameCurr_->setPose(estimatedPoseCurr_);

            // Since the pose of frame is updated, try again to get more matches
            cout << "Fine computing...\n";
            matchKeyPointsWithActiveMapPoints();
            estimatePosePnP(true);
            frameCurr_->setPose(estimatedPoseCurr_);

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

            // remove non-active mappoints
            // cullNonActiveMapPoints();

            if (isKeyFrame())
            {
                cout << "  Current frame is a new keyframe" << endl;
                MapManager::GetInstance().InsertKeyframe(frameCurr_);

                // Add this keyframe as the observation of observed mappoints
                addKeyframeObservationToOldMapPoints();

                // Add some new mappoints since current frame is keyframe
                addNewMapPoints();

                // Optimize active mappoints since has a new keyframe
                // triangulateActiveMapPoints();

                // if have backend, use backend to optimize mappoints position and frame pose
                if (backend_)
                {
                    frameCurr_->updateConnectedKeyFrames();
                    backend_->optimizeCovisibilityGraph(frameCurr_);
                }

                framePrev_ = frameCurr_;
                keyframeRef_ = frame;
            }
            break;
        }
        case LOST:
        {
            cout << "Tracking has lost" << endl;
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

    void FrontEnd::extractKeyPointsAndComputeDescriptors()
    {
        orb_->detectAndCompute(frameCurr_->color_, Mat(), keypointsCurr_, descriptorsCurr_);
    }

    void FrontEnd::matchKeyPointsWithActiveMapPoints()
    {
        // Update the tracking map
        if (keyframeForTrackingMap_ != keyframeRef_) {
            keyframeForTrackingMap_ = keyframeRef_;
            trackingMap_ = MapManager::GetInstance().GetMappointsAroundKeyframe(keyframeRef_);
        }
        if (trackingMap_.size() < 100) {
            trackingMap_ = MapManager::GetInstance().GetAllMappoints();
            cout << " Not enough active mappoints, reset activie mappoints to all mappoints" << endl;
        }

        // Select the good mappoints candidates
        vector<MapPoint::Ptr> mptCandidates;
        Mat mptCandidatesDescriptors;
        for (auto &mappoint : trackingMap_)
        {
            auto mp = mappoint.second;

            // If considered as outlier by backend
            // TODO: should remove this mappoint from the trackingMap_
            if (mp->outlier_)
            {
                continue;
            }

            // If cannot be viewed by current frame
            // TODO: should remove this mappoint from the trackingMap_
            if (!frameCurr_->isInFrame(mp->getPosition()))
            {
                continue;
            }

            // add as a candidate
            mp->visibleTimes_++;
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

        matchedMptKptMap_.clear();
        matchedKptSet_.clear();
        for (cv::DMatch &m : matches)
        {
            // filter out the matches whose distance is large
            if (m.distance < max<float>(min_dis * minDisRatio_, 30.0))
            {
                matchedMptKptMap_[mptCandidates[m.queryIdx]] = keypointsCurr_[m.trainIdx];
                matchedKptSet_.insert(keypointsCurr_[m.trainIdx]);
            }
        }
        cout << "  Active mappoints size: " << trackingMap_.size() << endl;
        cout << "  Candidate mappoints size: " << mptCandidates.size() << endl;
        cout << "  Matched <mappoint, keypoint> paris size: " << matchedMptKptMap_.size() << endl;
    }

    void FrontEnd::estimatePosePnP(bool addObservation)
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

        // use PNP to compute the initial pose
        Mat initRotMat, rotVec, tranVec, inliers;
        cv::eigen2cv(frameCurr_->getPose().rotationMatrix(), initRotMat);
        cv::Rodrigues(initRotMat, rotVec);
        cv::eigen2cv(frameCurr_->getPose().translation(), tranVec);

        cv::solvePnPRansac(pts3d, pts2d, frameCurr_->camera_->getCameraMatrix(), Mat(),
                           rotVec, tranVec, true,
                           100, 4.0, 0.99,
                           inliers, cv::SOLVEPNP_P3P);
        num_inliers_ = inliers.rows;
        cout << "  PNP results inlier size: " << num_inliers_ << endl;

        // Covert rotation vector to matrix
        Mat rotMat;
        cv::Rodrigues(rotVec, rotMat);
        Eigen::Matrix3d rotMat_eigen;
        rotMat_eigen << rotMat.at<double>(0, 0), rotMat.at<double>(0, 1), rotMat.at<double>(0, 2),
            rotMat.at<double>(1, 0), rotMat.at<double>(1, 1), rotMat.at<double>(1, 2),
            rotMat.at<double>(2, 0), rotMat.at<double>(2, 1), rotMat.at<double>(2, 2);
        estimatedPoseCurr_ = SE3(
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
        pose->setEstimate(estimatedPoseCurr_);
        optimizer.addVertex(pose);

        // edges
        vector<UnaryEdgeProjection *> edges;

        for (size_t i = 0; i < inliers.rows; i++)
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
        for (size_t i = 0; i < edges.size(); i++)
        {
            auto edge = edges[i];
            edge->computeError();

            if (edge->chi2() > 1)
            {
                edge->setLevel(1);
                outlierCnt++;
            }
            else
            {
                auto mpt = mpts3d[inliers.at<int>(i, 0)];
                mpt->matchedTimes_++;
                if (addObservation) {
                    frameCurr_->addObservedMapPoint(mpt);
                }
            }
            edge->setRobustKernel(0);
        }
        cout << "got outliders after first round BA: " << outlierCnt << endl;

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        estimatedPoseCurr_ = pose->estimate();

        // TODO: remove the outliers from active map?
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
        SE3 T_r_c = framePrev_->getPose() * estimatedPoseCurr_.inverse();
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
        SE3 T_r_c = framePrev_->getPose() * estimatedPoseCurr_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > keyFrameMinRot_ || trans.norm() > keyFrameMinTrans_)
        {
            return true;
        }
        return false;
    }

    void FrontEnd::addNewMapPoints()
    {
        for (size_t i = 0; i < keypointsCurr_.size(); i++)
        {
            // if the keypoint doesn't match with previous mappoints, this is a new mappoint
            if (!matchedKptSet_.count(keypointsCurr_[i]))
            {
                addNewMapPoint(i);
            }
        }
        // MapManager::GetInstance().updateMappointEraseRatio();
        // cout << "  Active mappoints size after adding: " << MapManager::GetInstance().getActiveMappoints().size() << endl;
    }

    void FrontEnd::addNewMapPoint(int idx)
    {
        double depth = frameCurr_->findDepth(keypointsCurr_[idx]);
        if (depth < 0)
        {
            return;
        }

        Vector3d mptPos = frameCurr_->camera_->pixel2world(
            keypointsCurr_[idx], frameCurr_->getPose(), depth);

        // Create a mappoint
        // deep copy the descriptor and keypoint position since they will be clear in next loop
        MapPoint::Ptr mpt = MapPoint::createMapPoint(
            mptPos,
            (mptPos - frameCurr_->getCamCenter()).normalized(),
            descriptorsCurr_.row(idx).clone(),
            frameCurr_->getId(),
            cv::Point2f(keypointsCurr_[idx].pt));

        // set this mappoint as the observed mappoints of current frame
        frameCurr_->addObservedMapPoint(mpt);

        // Add mappoint into map
        MapManager::GetInstance().InsertMappoint(mpt);
    }

    void FrontEnd::addKeyframeObservationToOldMapPoints()
    {
        for (auto &mappoint : frameCurr_->getObservedMapPoints())
        {
            if (mappoint.expired())
            {
                continue;
            }

            auto mp = mappoint.lock();
            if (mp->outlier_)
            {
                continue;
            }

            mp->addKeyFrameObservation(frameCurr_->getId(), cv::Point2f(matchedMptKptMap_[mp].pt));
        }
    }

    // void FrontEnd::triangulateActiveMapPoints()
    // {
    //     int triangulatedCnt = 0;
    //     for (auto &mappoint : MapManager::GetInstance().getActiveMappoints())
    //     {
    //         auto mp = mappoint.second;
    //         if (mp->outlier_ || mp->triangulated_ || mp->optimized_)
    //         {
    //             continue;
    //         }

    //         // it current keyframe doesn't observe this mappoint, it cannot be triangulated
    //         if (!matchedMptKptMap_.count(mp))
    //         {
    //             continue;
    //         }

    //         // try to triangulate the mappoint
    //         vector<SE3> poses;
    //         vector<Vec3> points;
    //         for (auto &keyFrameMap : mp->getKeyFrameObservationsMap())
    //         {
    //             auto keyFrame = MapManager::GetInstance().GetKeyframe(keyFrameMap.first);
    //             auto keyPoint = keyFrameMap.second;

    //             if (keyFrame == nullptr)
    //             {
    //                 continue;
    //             }

    //             poses.push_back(keyFrame->getPose());
    //             points.push_back(keyFrame->camera_->pixel2camera(keyPoint));
    //         }

    //         if (poses.size() >= 2)
    //         {
    //             Vec3 pworld = Vec3::Zero();
    //             if (triangulation(poses, points, pworld) && pworld[2] > 0)
    //             {
    //                 // if triangulate successfully
    //                 mp->setPosition(pworld);
    //                 mp->triangulated_ = true;
    //                 triangulatedCnt++;
    //                 break;
    //             }
    //         }
    //     }
    //     cout << "  Triangulate active mappoints size: " << triangulatedCnt << endl;
    // }

} // namespace