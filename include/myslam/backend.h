/*
 * Backend for covisible graph (tracking map in frontend) optimization
 */

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

    Backend(const Camera::Ptr camera): camera_(move(camera)) {
        backendRunning_ = true;
        backendThread_ = std::thread(std::bind(&Backend::BackendLoop, this));
        chi2Threshold_ = Config::get<float>("chi2_th");
    }

    void Stop() {
        backendRunning_ = false;
        backendUpdateTrigger_.notify_one();
        backendThread_.join();
    }

    void OptimizeCovisibleGraphOfKeyframe(const Frame::Ptr keyframeCurr) {
        unique_lock<mutex> lock(backendMutex_);
        keyframeCurr_ = move(keyframeCurr);
        backendUpdateTrigger_.notify_one();
    }

private:

    thread              backendThread_;
    bool                backendRunning_;
    mutex               backendMutex_;
    condition_variable  backendUpdateTrigger_;

    Camera::Ptr         camera_;
    Frame::Ptr          keyframeCurr_;

    float               chi2Threshold_;

    // main function for backend thread
    void BackendLoop();

    // real perform the optimization
    void Optimize();
}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
