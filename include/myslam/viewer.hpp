#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <myslam/common_include.hpp>

#include <pangolin/pangolin.h>

#include <mutex>
#include <thread>
#include <unordered_set>

namespace myslam {

class Frame;
class Mappoint;

class Viewer {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer() {
        viewer_running_ = true;
        // viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
        Setup();
    }

    void SingleStep();

    void Stop() {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void SetCurrentFrame(
        const cv::Mat& colorImage,
        const std::shared_ptr<Frame>& current_frame, 
        const std::unordered_set<size_t>& matchedKptsIdx);

    /*
      Update the all_keyframes_ and all_mappoints_
    */
    void UpdateDrawingObjects();

private:
    bool viewer_running_;
    std::thread viewer_thread_;
    std::mutex viewer_data_mutex_;

    std::unordered_map<size_t, std::shared_ptr<Frame>> all_keyframes_;
    std::unordered_map<size_t, std::shared_ptr<Mappoint>> all_mappoints_;
    std::unordered_map<size_t, std::shared_ptr<Mappoint>> active_mappoints_;
    cv::Mat colorImage_;
    std::shared_ptr<Frame> current_frame_;

    std::unordered_set<size_t> matchedKptsIdx_;

    pangolin::OpenGlRenderState vis_camera_;
    pangolin::View vis_display_;

    void ThreadLoop();

    void Setup();

    void DrawFrame(std::shared_ptr<Frame> frame, const float* color);

    void DrawMapPoints();

    void DrawOtherKeyFrames();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    void PlotFrameImage();

    // Get keypoint color
    cv::Scalar GetKeypointColor(size_t kptIdx);

}; // class Viewer

} // namespace


#endif  // MYSLAM_VIEWER_H