#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/mapmanager.h"
#include "myslam/camera.h"
#include "myslam/frame.h"

namespace myslam {

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend() {
        backendRunning_ = true;
        backendThread_ = std::thread(std::bind(&Backend::backendLoop, this));
        chi2_th_ = Config::get<float>("chi2_th");
    }

    void Stop() {
        backendRunning_ = false;
        mapUpdate_.notify_one();
        backendThread_.join();
    }

    void optimizeCovisibilityGraph(const Frame::Ptr& keyFrameCurr) {
        unique_lock<mutex> lock(backendMutex_);
        keyFrameCurr_ = keyFrameCurr;
        mapUpdate_.notify_one();
    }

    void setCamera(const Camera::Ptr& camera) { camera_ = camera; }

private:

    bool backendRunning_;
    thread backendThread_;
    mutex backendMutex_;
    condition_variable mapUpdate_;
    void backendLoop();

    void optimize();

    Camera::Ptr camera_;
    Frame::Ptr keyFrameCurr_;

    float chi2_th_;

}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
