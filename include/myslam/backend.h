#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/map.h"

namespace myslam {

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend() {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }

    void Stop() {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void OptimizeMap() {
        unique_lock<mutex> lock(backend_mutex_);
        map_update_.notify_one();
    }

    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

private:

    atomic<bool> backend_running_;
    thread backend_thread_;
    mutex backend_mutex_;
    condition_variable map_update_;
    void BackendLoop();

    void Optimize(Map::KeyframeDict& keyframes, Map::MappointDict& mappoints);


    std::shared_ptr<Map> map_;


}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
