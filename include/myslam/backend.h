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

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/frame.h"
#include "myslam/g2o_types.h"

namespace myslam {

struct FrontendToBackendInfo {
    Frame::Ptr                              keyframe;
    unordered_map<size_t, size_t>           oldMptIdKptIdxMap;
    unordered_map<Mappoint::Ptr, size_t>    newMptKptIdxMap;
};

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    typedef std::shared_ptr<Backend> Ptr;

    Backend(const Camera::Ptr camera);

    void RegisterTrackingMapUpdateCallback(function<void(function<void(unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler) {
        frontendMapUpdateHandler_ = frontendMapUpdateHandler;
    }

    // Stop backend processing and clean up resources
    void Stop();

    // Add a new keyframe to the queue
    void AddNewKeyframeInfo(const FrontendToBackendInfo& info);

private:

    thread              backendThread_;
    bool                backendRunning_;
    mutex               backendMutex_;
    condition_variable  backendUpdateTrigger_;

    Camera::Ptr         camera_;

    Frame::Ptr                              keyframePrev_;
    Frame::Ptr                              keyframeCurr_;
    unordered_map<size_t, size_t>           oldMptIdKptIdxMap_;
    unordered_map<Mappoint::Ptr, size_t>    newMptKptIdxMap_;
    
    queue<FrontendToBackendInfo>            frontendInfoToProcess_;

    double                                  reMatchDescriptorDistance_;
    
    g2o::SparseOptimizer                    optimizer_;
    double                                  chi2Threshold_;

    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>>                    kfIdToCovKfThenVertex_;
    unordered_map<size_t, pair<Mappoint::Ptr, VertexMappoint*>>             mptIdToMptThenVertex_;
    // keyframes not belonging to covisible keyframes but could observe the local mappoints
    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>>                    kfIdToFixedKfThenVertex_;
    unordered_map<BinaryEdgeProjection*, pair<Frame::Ptr, Mappoint::Ptr>>   edgeToKfThenMpt_;

    list<pair<Frame::Ptr, size_t>>  observingMptToRemove_;
    unordered_set<Mappoint::Ptr>    observingMptToRemoveSet_;

    function<void(function<void(unordered_map<size_t, Mappoint::Ptr>&)>)> frontendMapUpdateHandler_;

    // main function for backend thread
    void BackendLoop();

    // get the info from the queue
    void PopInfoFromQueue();

    // project more existing mappoint to new keyframe
    void ProjectMoreMappointsToNewKeyframe();

    // add observing mappoints, including old and new, to new keyframe
    void AddObservingMappointsToNewKeyframe();

    // add the new observations for old keyframes
    void AddNewMappointsToExistingKeyframe();

    // perform the optimization for local map
    void OptimizeLocalMap();

    // update frontend tracking map
    void UpdateFrontendTrackingMap();

    // clean up the allocated memory
    void CleanUp();

}; // class Backend

} // namespace

#endif  // MYSLAM_BACKEND_H
