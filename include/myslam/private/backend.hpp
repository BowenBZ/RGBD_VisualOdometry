/*
 * Backend for covisible graph (tracking map in frontend) optimization
 *
 * Backend is reponsible for the other functions
 * 1. update covisible graph for new keyframe and new mappoints
 * 2. generate more observations for new mappoints and merge mappoints
 * 3. bundle adjustment for covisible graph of new keyframe
 * 4. update the tracking map for frontend
 */

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include <myslam/common_include.hpp>
#include <myslam/camera.hpp>

#include "myslam/private/frame.hpp"
#include "myslam/private/g2o_types.hpp"
#include "myslam/private/mapmanager.hpp"
#include "myslam/private/mappoint.hpp"

#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace myslam {

typedef struct {
    double  reMatchDescriptorDistance;
    double  baInlierThres;
} BackendConfig;

typedef struct {
    BinaryEdgeProjection* edge;
    bool isOutlier;
    Frame::Ptr keyframe;
    Mappoint::Ptr mappoint;
} GraphEdgeInfo;

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    typedef std::shared_ptr<Backend> Ptr;

    Backend(const Camera::Ptr camera);

    void RegisterTrackingMapUpdateCallback(std::function<void(std::function<void(std::unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler) {
        frontendMapUpdateHandler_ = frontendMapUpdateHandler;
    }

    // Stop backend processing and clean up resources
    void Stop();

    // Add a new keyframe to the queue
    void AddNewKeyframeInfoToQueue(const size_t keyframeId);

    // If backend is idle
    bool IsIdle() const {
        return isIdle_;
    }

private:
    BackendConfig       config_;

    std::thread         backendThread_;
    bool                backendRunning_;
    std::mutex          backendMutex_;
    std::condition_variable  backendUpdateTrigger_;

    Camera::Ptr         camera_;
    MapManager*         mapManager_;

    bool                isIdle_;

    // Queue to store new keyframe id from frontend
    std::queue<size_t>  newKeyframeIdQueue_;

    Frame::Ptr          keyframePrev_;
    Frame::Ptr          keyframeCurr_;
    
    g2o::SparseOptimizer                                                    optimizer_;

    std::unordered_map<size_t, std::pair<Frame::Ptr, VertexPose*>>                    kfIdToCovKfThenVertex_;
    std::unordered_map<size_t, std::pair<Mappoint::Ptr, VertexMappoint*>>             mptIdToMptThenVertex_;
    // keyframes not belonging to covisible keyframes but could observe the local mappoints
    std::unordered_map<size_t, std::pair<Frame::Ptr, VertexPose*>>                    kfIdToFixedKfThenVertex_;
    std::list<GraphEdgeInfo>                                                     edges_;

    std::list<std::pair<Frame::Ptr, size_t>>                                          observingMptToRemove_;
    
    // New created mappoints from current keyframe needs to be removed if we found previous matched mappoint
    std::list<size_t>                                                            mptIdToRemove_;

    std::function<void(std::function<void(std::unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler_;

    // main function for backend thread
    void BackendLoop();

    // get the info from the queue
    void PopInfoFromQueue();

    // project more existing mappoint to new keyframe
    void ProjectMoreMappointsToNewKeyframe();

    // perform the optimization for local map
    void OptimizeLocalMap();

    // update frontend tracking map
    void UpdateFrontendTrackingMap();

    // clean up the allocated memory
    void CleanUp();

}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
