#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/camera.h"

namespace myslam {

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend() {
        backendRunning_ = true;
        backendThread_ = std::thread(std::bind(&Backend::backendLoop, this));
    }

    void Stop() {
        backendRunning_ = false;
        mapUpdate_.notify_one();
        backendThread_.join();
    }

    void optimizeActivePosesAndActiveMapPoints() {
        unique_lock<mutex> lock(backendMutex_);
        mapUpdate_.notify_one();
    }

    void setMap(const Map::Ptr& map) { map_ = map; }

    void setCamera(const Camera::Ptr& camera) { camera_ = camera; }

private:

    bool backendRunning_;
    thread backendThread_;
    mutex backendMutex_;
    condition_variable mapUpdate_;
    void backendLoop();

    void optimize(Map::KeyframeDict& keyframes, Map::MappointDict& mappoints);

    Map::Ptr map_;
    Camera::Ptr camera_;

}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
