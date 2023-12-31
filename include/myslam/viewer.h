#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pangolin/pangolin.h>
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/util.h"
#include "myslam/mapmanager.h"
#include <opencv2/features2d/features2d.hpp>

namespace myslam {

class Viewer {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer() {
        viewer_running_ = true;
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    }

    void Stop() {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void SetCurrentFrame(
        const Frame::Ptr& current_frame, 
        const unordered_set<size_t>& matchedKptsIdx,
        const unordered_set<size_t>& inlierKptsIdx);

    /*
      Update the all_keyframes_ and all_mappoints_
    */
    void UpdateDrawingObjects();

private:
    bool viewer_running_;
    thread viewer_thread_;
    mutex viewer_data_mutex_;

    MapManager::KeyframeIdToPtr all_keyframes_;
    MapManager::MappointIdToPtr all_mappoints_;
    MapManager::MappointIdToPtr active_mappoints_;
    Frame::Ptr current_frame_;
    KeyPointSet keypointsCurr_;

    unordered_set<size_t> matchedKptsIdx_;
    unordered_set<size_t> inlierKptsIdx_;

    void ThreadLoop();

    void DrawFrame(Frame::Ptr frame, const float* color);

    void DrawMapPoints();

    void DrawOtherKeyFrames();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    // Get keypoint color
    cv::Scalar GetKeypointColor(size_t kptIdx);

}; // class Viewer

} // namespace


#endif  // MYSLAM_VIEWER_H