/*
 * Backend for covisible graph (tracking map in frontend) optimization
 */

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/mapmanager.h"
#include "myslam/camera.h"
#include "myslam/frame.h"
#include "myslam/g2o_types.h"

namespace myslam {

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    typedef std::shared_ptr<Backend> Ptr;

    Backend(const Camera::Ptr camera);

    void RegisterTrackingMapUpdateCallback(function<void(function<void(Frame::Ptr&, unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler) {
        frontendMapUpdateHandler_ = frontendMapUpdateHandler;
    }

    // Stop backend processing and clean up resources
    void Stop();

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
    
    g2o::SparseOptimizer optimizer_;

    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>> kfIdToCovKfThenVertex_;
    unordered_map<size_t, pair<Mappoint::Ptr, VertexMappoint*>> mptIdToMptThenVertex_;
    // keyframes not belonging to covisible keyframes but could observe the local mappoints
    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>> kfIdToFixedKfThenVertex_;
    unordered_map<BinaryEdgeProjection*, pair<Frame::Ptr, Mappoint::Ptr>> edgeToKfThenMpt_;

    function<void(function<void(Frame::Ptr&, unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler_;

    // main function for backend thread
    void BackendLoop();

    // real perform the optimization
    void Optimize();

    // update frontend tracking map
    void UpdateFrontendTrackingMap();

    // Clean up the allocated memory
    void CleanUp();

}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
