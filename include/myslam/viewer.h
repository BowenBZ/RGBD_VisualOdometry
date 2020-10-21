#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pangolin/pangolin.h>
#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/frame.h"

namespace myslam {

class Viewer {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer() {
        viewer_running_ = true;
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    }
    
    void SetMap(Map::Ptr map) { map_ = map; }

    void Close() {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void SetCurrentFrame(Frame::Ptr current_frame) {
        unique_lock<mutex> lck(viewer_data_mutex_);
        current_frame_ = current_frame;
    }

    /*
      Update the all_keyframes_ and all_mappoints_
    */
    void UpdateMap();

private:
    bool viewer_running_;
    thread viewer_thread_;
    mutex viewer_data_mutex_;

    Map::Ptr map_ = nullptr;
    Map::KeyframeDict all_keyframes_;
    Map::MappointDict all_mappoints_;
    Frame::Ptr current_frame_;

    void ThreadLoop();

    void DrawFrame(Frame::Ptr frame, const float* color);

    void DrawMapPoints();

    void DrawOtherKeyFrames();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

}; // class Viewer

} // namespace


#endif  // MYSLAM_VIEWER_H