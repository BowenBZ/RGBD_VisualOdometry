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

FrontEnd::FrontEnd() :
    state_ ( INITIALIZING ), frameRef_ ( nullptr ), frameCurr_ ( nullptr ), num_lost_ ( 0 ), num_inliers_ ( 0 ),
    flannMatcher_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    orb_ = cv::ORB::create ( Config::get<int> ( "number_of_features" ), 
                             Config::get<double> ( "scale_factor" ), 
                             Config::get<int> ( "level_pyramid" ) );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );

    cout << "Frontend status: -1: Initialization, 0: Tracking, 1: Lost" << endl;
}

bool FrontEnd::addFrame ( Frame::Ptr frame )
{
    cout << "Frontend status: " << state_ << endl;
    frameCurr_ = frame;

    switch ( state_ )
    {
    case INITIALIZING:
    {
        // RGBD camera only needs 1 frame to configure since it could get the depth information
        state_ = TRACKING;

        // extract features from frameCurr_
        extractKeyPointsAndComputeDescriptors();

        // the first frame is a key-frame
        map_->insertKeyFrame ( frameCurr_ );
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

        if ( isGoodEstimation() )
        {
            // update to estimated pose
            frameCurr_->setPose(estimatedPoseCurr_); 
            // remove non-active mappoints, may add new mappoints if size is too small
            updateActiveMapPointsMap(); 
            num_lost_ = 0;
            if ( isKeyFrame() )
            {
                cout << "  Is a new keyframe" << endl;
                map_->insertKeyFrame ( frameCurr_ );
                optimizeActiveMapPointsPosition();
                frameRef_ = frameCurr_;
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"Tracking has lost."<<endl;
        break;
    }
    }

    viewer_->setCurrentFrame ( frameCurr_, matchedKptSet_ );
    viewer_->updateDrawingObjects();

    return true;
}

void FrontEnd::extractKeyPointsAndComputeDescriptors()
{
    orb_->detectAndCompute( frameCurr_->color_, Mat(), keypointsCurr_, descriptorsCurr_ );
}

void FrontEnd::matchKeyPointsWithActiveMapPoints()
{
    // get the active mappoints candidates from map 
    auto activeMpts = map_->getActiveMappoints();
    if (activeMpts.size() < 100) {
        activeMpts = map_->getAllMappoints();
        map_->resetActiveMappoints();
        cout << " Not enough active mappoints, reset activie mappoints to all mappoints" << endl;
    }

    // Select the good mappoints candidates
    vector<MapPoint::Ptr> mptCandidates;
    Mat descriptorCandidates;
    for ( auto& mappoint: activeMpts )
    {
        auto mp = mappoint.second;

        if(mp->outlier_)
            continue;

        if ( frameCurr_->isInFrame( mp->getPosition() ) )
        {
            // add to candidate 
            mp->visible_times_++;
            mptCandidates.push_back( mp );
            descriptorCandidates.push_back( mp->descriptor_ );
        }
    }
    
    vector<cv::DMatch> matches;
    flannMatcher_.match ( descriptorCandidates, descriptorsCurr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), 
                        matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
                        { return m1.distance < m2.distance; } 
                        ) 
                        -> distance;

    matchedMptKptMap_.clear();
    matchedKptSet_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            matchedMptKptMap_[mptCandidates[m.queryIdx]] = keypointsCurr_[m.trainIdx];
            matchedKptSet_.insert( keypointsCurr_[m.trainIdx] );
        }
    }
    cout<<"  Active mappoints size: " << mptCandidates.size() << endl;
    cout<<"  Matched feature paris size: "<<matchedMptKptMap_.size() <<endl;
}

void FrontEnd::estimatePosePnP()
{
    // construct the 3d 2d observations
    vector<MapPoint::Ptr> mpts3d;
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    for ( auto& pair: matchedMptKptMap_ )
    {
        mpts3d.push_back( pair.first );
        pts3d.push_back( toPoint3f(pair.first->getPosition()) ); 
        pts2d.push_back( pair.second.pt );
    }
    
    // use P3P wih Ransac to solve an intial value of the pose
    Mat K = frameCurr_->camera_->getCameraMatrix();
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_P3P );

    num_inliers_ = inliers.rows;
    cout<<"  PNP results inlier size: "<<num_inliers_<<endl;
    Mat R;
    cv::Rodrigues(rvec, R); // Covert rotation vector to matrix
    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    estimatedPoseCurr_ = SE3(
        R_eigen,
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );

    // using bundle adjustment to optimize the pose 
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    
    VertexPose* pose = new VertexPose();
    pose->setId ( 0 );
    pose->setEstimate ( estimatedPoseCurr_ );
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        UnaryEdgeProjection* edge = new UnaryEdgeProjection(toVec3d(pts3d[index]), frameCurr_->camera_);
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->setMeasurement( toVec2d(pts2d[index]) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
        // set the inlier map points 
        mpts3d[index]->matched_times_++;
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    estimatedPoseCurr_ = pose->estimate();
}

bool FrontEnd::isGoodEstimation()
{
    // check if inliers number meet the threshold
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"Current tracking is rejected because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // check if the motion is too large
    SE3 T_r_c = frameRef_->getPose() * estimatedPoseCurr_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"Current tracking is rejected because motion is too large: "<<d.norm()<<endl;
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
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void FrontEnd::initMap() {

    for ( size_t i=0; i < keypointsCurr_.size(); i++ )
    {
        double depth = frameCurr_->findDepth ( keypointsCurr_[i] );
        if ( depth < 0 ) {
            continue;
        }
        Vector3d p_world = frameCurr_->camera_->pixel2world (
            keypointsCurr_[i], 
            frameCurr_->getPose(), 
            depth
        );
        Vector3d direction = p_world - frameCurr_->getCamCenter();
        direction.normalize();
        MapPoint::Ptr mpt = MapPoint::createMapPoint(
            p_world, direction, keypointsCurr_[i].pt, descriptorsCurr_.row(i).clone(), frameCurr_
        );
        map_->insertMapPoint( mpt );
    }
}

void FrontEnd::addNewMapPoints()
{
    for ( int i = 0; i < keypointsCurr_.size(); i++ )
    {
        cv::KeyPoint keypoint = keypointsCurr_[i];
        // if the keypoint doesn't match with previous mappoints, so this is a new mappoint
        if ( !matchedKptSet_.count(keypoint) )   
        {
            double d = frameCurr_->findDepth ( keypoint );
            if ( d < 0 )  
                continue;

            Vector3d p_world = frameCurr_->camera_->pixel2world (
                keypoint,
                frameCurr_->getPose(), 
                d
            );
            Vector3d n = p_world - frameCurr_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, keypoint.pt, descriptorsCurr_.row(i).clone(), frameCurr_
            );
            map_->insertMapPoint( map_point );
        }
    }
}

void FrontEnd::updateActiveMapPointsMap()
{
    map_->cullNonActiveMapPoints( frameCurr_ );

    if ( matchedMptKptMap_.size()  < 100 ) {
        addNewMapPoints();
    }

    map_->updateMappointEraseRatio();
    
    cout<<"  Active mappoints size after updating: "<<map_->getActiveMappoints().size()<<endl;
}

void FrontEnd::optimizeActiveMapPointsPosition() {
    int triangulate_cnt = 0;
    for (auto& mappoint : map_->getActiveMappoints()) {
        auto mp = mappoint.second;
        // Check whether this mappoint have matches
        if (mp->triangulated_ || !matchedMptKptMap_.count(mp)) {
            continue;
        }

        auto lastKeyFrameObservationMap = mp->getKeyFrameObservationsMap().back();
        auto lastKeyFrame = lastKeyFrameObservationMap.first.lock();

        // the first observed frame may not be keyframe, so this weak_ptr may point to null
        if(!lastKeyFrame) {
            mp -> addKeyFrameObservation(frameCurr_, matchedMptKptMap_[mp].pt);
            continue;
        }

        cv::Point2f lastPt = lastKeyFrameObservationMap.second;
        cv::Point2f currPt = matchedMptKptMap_[mp].pt;
        vector<SE3> poses {lastKeyFrame->getPose(), frameCurr_->getPose()};
        vector<Vec3> points {lastKeyFrame->camera_->pixel2camera(lastPt), 
                             frameCurr_->camera_->pixel2camera(currPt)};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0)
        {
            // if triangulate successfully
            mp->setPosition(pworld);
            mp->triangulated_ = true;
            triangulate_cnt++;
        }

        mp->addKeyFrameObservation(frameCurr_, currPt);
    }
    cout << "  Triangulate active mappoints size: " << triangulate_cnt << endl;

    // if have backend, use backend to optimize mappoints position and frame pose
    if(backend_) {
        backend_->optimizeActivePosesAndActiveMapPoints();
    }
}

} //namespace