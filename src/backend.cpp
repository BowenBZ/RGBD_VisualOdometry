#include "myslam/backend.h"
#include "myslam/g2o_types.h"

namespace myslam {

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        unique_lock<mutex> lock(backend_mutex_);
        map_update_.wait(lock);

        auto active_kfs = map_->getActiveKeyFrames();
        auto active_mappoints = map_->getActiveMappoints();
        Optimize(active_kfs, active_mappoints);
    }
}

void Backend::Optimize(Map::KeyframeDict &keyframes,
                       Map::MappointDict &mappoints) {


}


} // namespace 