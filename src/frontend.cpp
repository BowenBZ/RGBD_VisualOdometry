/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
#include "myslam/algorithm.h"

namespace myslam
{

FrontEnd::FrontEnd() :
    state_ ( INITIALIZING ), ref_frame_ ( nullptr ), curr_frame_ ( nullptr ), num_lost_ ( 0 ), num_inliers_ ( 0 ),
    matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    orb_ = cv::ORB::create ( Config::get<int> ( "number_of_features" ), 
                             Config::get<double> ( "scale_factor" ), 
                             Config::get<int> ( "level_pyramid" ) );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
}

FrontEnd::~FrontEnd()
{

}

bool FrontEnd::addFrame ( Frame::Ptr frame )
{
    cout << "Current status: " << (int)state_ << endl;
    curr_frame_ = frame;

    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = TRACKING;

        // extract features from curr_frame_
        extractKeyPoints();
        computeDescriptors();

        // the first frame is a key-frame
        map_->insertKeyFrame ( curr_frame_ );
        initMap();
        ref_frame_ = frame;
        break;
    }
    case TRACKING:
    {
        // set a initial pose, used for looking for whether map points are in frame
        curr_frame_->T_c_w_ = ref_frame_->T_c_w_;   

        extractKeyPoints();
        computeDescriptors();

        // Map the keypoints of current frames with the points in map 
        featureMatching();
        // Estimate pose of current frame
        poseEstimationPnP();

        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_frame_->T_c_w_ = T_c_w_estimated_;  // update to estimated pose
            cullNonActiveMapPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                map_->insertKeyFrame ( curr_frame_ );
                optimizeActiveMapPoints();
                ref_frame_ = curr_frame_;
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
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    viewer_->SetCurrentFrame ( curr_frame_, match_2dpts_ );
    viewer_->UpdateMap();

    return true;
}

void FrontEnd::extractKeyPoints()
{
    orb_->detect ( curr_frame_->color_, keypoints_curr_ );
}

void FrontEnd::computeDescriptors()
{
    orb_->compute ( curr_frame_->color_, keypoints_curr_, descriptors_curr_ );
}

void FrontEnd::featureMatching()
{
    vector<cv::DMatch> matches;
    Mat desp_map;
    // get the active candidates from map 
    vector<MapPoint::Ptr> candidate;
    for ( auto& mappoint: map_->getActiveMappoints() )
    {
        auto p = mappoint.second;
        // check if p in curr frame image 
        // if ( curr_frame_->isInFrame( p->pos_ ) )
        // {
            // add to candidate 
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        // }
    }
    
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    cout << "matches size: " << matches.size() << endl;
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_3d_2d_pts_.clear();
    match_2dkp_index_.clear();
    match_2dpts_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_3d_2d_pts_[candidate[m.queryIdx]] = keypoints_curr_[m.trainIdx].pt;
            match_2dkp_index_.insert( m.trainIdx );
            match_2dpts_.push_back (keypoints_curr_[m.trainIdx].pt );
        }
    }
    cout<<"good matches: "<<match_3d_2d_pts_.size() <<endl;
}

void FrontEnd::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() ); 
        pts2d.push_back( match_3d_2d_pts_[pt] );
    }
    
    Mat K = curr_frame_->camera_->getCameraMatrix();
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );

    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    Mat R;
    cv::Rodrigues(rvec, R); // Covert rotation vector to matrix
    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    T_c_w_estimated_ = SE3(
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
    pose->setEstimate ( T_c_w_estimated_ );
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        EdgeProjection* edge = new EdgeProjection(Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z ), curr_frame_->camera_.get());
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->setMeasurement( toVec2d(pts2d[index]) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    T_c_w_estimated_ = pose->estimate();
}

bool FrontEnd::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_frame_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool FrontEnd::checkKeyFrame()
{
    SE3 T_r_c = ref_frame_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void FrontEnd::initMap() {

    for ( size_t i=0; i < keypoints_curr_.size(); i++ )
    {
        double d = curr_frame_->findDepth ( keypoints_curr_[i] );
        if ( d < 0 ) 
            continue;
        Vector3d p_world = curr_frame_->camera_->pixel2world (
            toVec2d ( keypoints_curr_[i]), curr_frame_->T_c_w_, d
        );
        Vector3d n = p_world - curr_frame_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, keypoints_curr_[i].pt, descriptors_curr_.row(i).clone(), curr_frame_
        );
        map_->insertMapPoint( map_point );
    }
}

void FrontEnd::addMapPoints()
{
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        // if the keypoint doesn't match with previous mappoints, so this is a new mappoint
        if ( !match_2dkp_index_.count(i) )   
        {
            double d = curr_frame_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 )  
                continue;
            Vector3d p_world = curr_frame_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
                curr_frame_->T_c_w_, d
            );
            Vector3d n = p_world - curr_frame_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, keypoints_curr_[i].pt, descriptors_curr_.row(i).clone(), curr_frame_
            );
            map_->insertMapPoint( map_point );
        }
    }
}

void FrontEnd::cullNonActiveMapPoints()
{
    // remove the hardly seen and no visible points from active mappoints
    list<unsigned long> remove_id;
    for (auto& mappoint : map_->getActiveMappoints()) {
        auto mp = mappoint.second;

        // if not in current view
        if ( !curr_frame_->isInFrame(mp->pos_) ) {
            remove_id.push_back(mappoint.first);
            continue;
        }

        // not often matches
        float match_ratio = float(mp->matched_times_) / mp->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            remove_id.push_back(mappoint.first);
            continue;
        }

        // not in good view
        double angle = getViewAngle( curr_frame_, mp );
        if ( angle > M_PI/6. )
        {
            remove_id.push_back(mappoint.first);
            continue;
        }
    }

    map_->removeActiveMapPoints(remove_id);

    if ( match_3d_2d_pts_.size()  < 100 ) {
        addMapPoints();
    }

    if ( map_->getActiveMappoints().size() > 1000 )  
    {
        // TODO map is too large, remove some one 
        map_point_erase_ratio_ += 0.05;
    }
    else {
        map_point_erase_ratio_ = 0.1;
    }
    
    cout<<"total active points: "<<map_->getActiveMappoints().size()<<endl;
}

double FrontEnd::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

void FrontEnd::optimizeActiveMapPoints() {
    for (auto& mappoint : map_->getActiveMappoints()) {
        auto mp = mappoint.second;
        // Check whether this mappoint have matches
        if (!mp->good_ && match_3d_2d_pts_.count(mp)) {
            auto pre_frame = mp->observed_frames_.back().lock();
            if(!pre_frame)
                continue;

            cv::Point2f pre_pt = mp->observed_pixel_pos_.back();
            cv::Point2f cur_pt = match_3d_2d_pts_[mp];
            vector<SE3> poses {pre_frame->T_c_w_, curr_frame_->T_c_w_};
            vector<Vec3> points {pre_frame->camera_->pixel2camera(Vector2d(pre_pt.x, pre_pt.y)), 
                                    curr_frame_->camera_->pixel2camera(Vector2d(cur_pt.x, cur_pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                mp->pos_ = pworld;
                mp->good_ = true;
            }
            else
            {
                mp->observed_frames_.push_back(curr_frame_);
                mp->observed_pixel_pos_.push_back(cur_pt);
            }
        }
    }
}

} //namespace