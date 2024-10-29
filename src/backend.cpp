#include "myslam/private/backend.h"

#include "mappoint.h"
#include "myslam/private/util.h"
#include "myslam/private/mapmanager.h"

#include <boost/timer/timer.hpp>

namespace myslam
{

Backend::Backend(const Camera::Ptr camera): camera_(std::move(camera)) {
    config_.baInlierThres = Config::get<double>("backend.ba_inlier_threshold");
    config_.reMatchDescriptorDistance = Config::get<double>("backend.re_match_descriptor_distance");

    mapManager_ = MapManager::Ptr(&MapManager::Instance());

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<CSparseLinearSolverType>()));
    optimizer_.setAlgorithm(solver);
    
    backendRunning_ = true;
    backendThread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::Stop() {
    backendRunning_ = false;
    backendUpdateTrigger_.notify_one();
    backendThread_.join();

    CleanUp();
}

void Backend::AddNewKeyframeInfoToQueue(const size_t keyframeId) {
    unique_lock<mutex> lock(backendMutex_);

    newKeyframeIdQueue_.push(keyframeId);
    backendUpdateTrigger_.notify_one();
}

void Backend::BackendLoop()
{
    while (backendRunning_)
    {
        unique_lock<mutex> lock(backendMutex_);
        if (newKeyframeIdQueue_.size() == 0) {
            isIdle_ = true;
            backendUpdateTrigger_.wait(lock);
        }
        isIdle_ = false;

        // Need to check again because the trigger could also be triggered during deconstruction
        if (backendRunning_)
        {
            printf("[Backend] info queue size: %zu\n", newKeyframeIdQueue_.size());
            PopInfoFromQueue();
            printf("[Backend] starts processing new frame: %zu\n", keyframeCurr_->GetId());
            // unlock so that frontend could keep sending info
            lock.unlock();

            boost::timer::cpu_timer timer;

            // ProjectMoreMappointsToNewKeyframe();
            // ProjectNewMappointsToExistingKeyframe();

            OptimizeLocalMap();

            boost::timer::cpu_times elapsed_times(timer.elapsed());
            printf("[Backend] Time cost (ms): %f\n\n", (elapsed_times.user + elapsed_times.system) / pow(10.0, 6.0));
            UpdateFrontendTrackingMap();
            CleanUp();
        }
    }
}

void Backend::PopInfoFromQueue() {

    auto& keyframeId = newKeyframeIdQueue_.front();

    keyframePrev_ = keyframeCurr_;
    keyframeCurr_ = mapManager_->GetKeyframe(keyframeId);
    assert(keyframeCurr_);

    newKeyframeIdQueue_.pop();
}

void Backend::ProjectMoreMappointsToNewKeyframe() {
    if (keyframePrev_ == nullptr) {
        return;
    }

    unordered_map<size_t, pair<size_t, double>> kptIdxToMptIdAndDistance;
    unordered_map<size_t, Mappoint::Ptr> oldMptIdToMpt;
    mapManager_->GetMappointsNearKeyframe(keyframePrev_, oldMptIdToMpt);
    for(auto& [oldMptId, oldMpt]: oldMptIdToMpt) {
        // Check if this mpt already match with new keyframe
        if (keyframeCurr_->IsObservingMappoint(oldMptId)) {
            continue;
        }

        double distance;
        size_t kptIdx;
        bool mayObserveMpt;
        // Cannot find match
        if (!keyframeCurr_->SearchKeypointMatchCandidate(oldMpt, false, kptIdx, distance, mayObserveMpt) || distance > config_.reMatchDescriptorDistance) {
            continue;
        }

        // this kpt already has matched previous mappoint
        // TODO: check if this mappoint is new created
        if (keyframeCurr_->GetMatchedMappointIdForKeypoint(kptIdx).has_value()) {
            continue;
        }

        // there is other old mpt matched with this new kpt
        if (kptIdxToMptIdAndDistance.count(kptIdx) && kptIdxToMptIdAndDistance[kptIdx].second <= distance) {
            continue;
        }

        kptIdxToMptIdAndDistance[kptIdx] = make_pair(oldMptId, distance);
    }

    // Add the new observations
    for(auto& [kptIdx, oldMptIdAndDistance]: kptIdxToMptIdAndDistance) {
        auto& [oldMptId, _] = oldMptIdAndDistance;
        auto& oldMpt = oldMptIdToMpt[oldMptId];
        keyframeCurr_->AddObservingMappointCreatedFromOtherFrame(kptIdx, oldMpt);
    }

    printf("[Backend] Projected %zu old mpts to new keyframe\n", kptIdxToMptIdAndDistance.size());
}

void Backend::ProjectNewMappointsToExistingKeyframe() {
    list<size_t> allCovisibleKfIds;
    keyframeCurr_->GetAllCovisibleKfIds(allCovisibleKfIds);
    unordered_set<Frame::Ptr> covisibleKfs;
    for (auto& kfId: allCovisibleKfIds) {
        auto kf = mapManager_->GetKeyframe(kfId);
        if (kf == nullptr) {
            continue;
        }
        covisibleKfs.insert(kf);
        list<size_t> neighborAllCovisibleKfIds;
        kf->GetAllCovisibleKfIds(neighborAllCovisibleKfIds);
        for (auto& neighborKfId: neighborAllCovisibleKfIds) {
            auto neighborKf = mapManager_->GetKeyframe(kfId);
            if (neighborKf == nullptr) {
                continue;
            }
            covisibleKfs.insert(neighborKf);
        }
    }
    if (covisibleKfs.count(keyframeCurr_)) {
        covisibleKfs.erase(keyframeCurr_);
    }

    // across different keyframes. old mpt to be replace by a new mpt
    unordered_map<size_t, pair<size_t, double>> oldMptIdToNewMptIdAndDistance;
    // for each keyframe. kpt to be matched with a new mpt
    unordered_map<size_t, pair<Mappoint::Ptr, double>> kptIdxToMptAndDistance;
    // across different keyframes. all new observations to be added
    list<tuple<Frame::Ptr, Mappoint::Ptr, size_t>> observationsToAdd;
    
    double distance;
    size_t kptIdx;
    bool mayObserveMpt;

    for (auto& kf: covisibleKfs) {

        kptIdxToMptAndDistance.clear();
        
        for (auto& mptId: keyframeCurr_->GetMappointIdsOnlyObservedByThisFrame()) {
            auto mpt = mapManager_->GetMappoint(mptId);
            if (!kf->SearchKeypointMatchCandidate(mpt, false, kptIdx, distance, mayObserveMpt) || 
                distance > config_.reMatchDescriptorDistance) {
                continue;
            }

            // check if the old keyframe keypoint already has matched mappoint
            const auto& optMptId = kf->GetMatchedMappointIdForKeypoint(kptIdx);
            if (optMptId.has_value()) {
                // if a previous matched keypoint could be matched with several new mappoints, find the best one for it
                const size_t& oldMptId = optMptId.value();
                if (oldMptIdToNewMptIdAndDistance.count(oldMptId) && 
                    distance >= oldMptIdToNewMptIdAndDistance[oldMptId].second) {
                    continue;
                }
                oldMptIdToNewMptIdAndDistance[oldMptId] = make_pair(mpt->GetId(), distance);
            } else {
                // if a previous empty keypoint could be matched with several new mappoints, find the best one for it
                if (kptIdxToMptAndDistance.count(kptIdx) &&
                    distance >= kptIdxToMptAndDistance[kptIdx].second) {
                    continue;
                }
                kptIdxToMptAndDistance[kptIdx] = make_pair(mpt, distance);
            }
        }
        // record observations to add for this keyframe 
        for(auto& [kptIdx, mptAndDistance]: kptIdxToMptAndDistance) {
            auto& [mpt, _] = mptAndDistance;
            observationsToAdd.push_back(make_tuple(kf, mpt, kptIdx));
        }
    }

    // add new observations
    for (auto& [kf, mpt, kptIdx]: observationsToAdd) {
        kf->AddObservingMappointCreatedFromOtherFrame(kptIdx, mpt);
    }

    // replace the previous mpt with the new mpt
    for (auto& [oldMptId, newMptIdAndDistance]: oldMptIdToNewMptIdAndDistance) {
        auto& [newMptId, _] = newMptIdAndDistance;
        MapManager::Instance().ReplaceMappoint(oldMptId, newMptId);
    }

    printf("[Backend] Added new mappoint observations to old keyframes: %zu\n", observationsToAdd.size());
    printf("[Backend] Replace old mappoints with new one: %zu\n", oldMptIdToNewMptIdAndDistance.size());
}

void Backend::OptimizeLocalMap()
{
    list<size_t> covisibleKfIds;
    keyframeCurr_->GetActiveCovisibleKfIds(covisibleKfIds);
    // Add current keyframe
    covisibleKfIds.push_back(keyframeCurr_->GetId());

    int vertexIndex = 0;

    // Create pose vertices and mappoint vertices for covisible keyframes
    for (auto &kfId : covisibleKfIds)
    {
        auto kf = mapManager_->GetKeyframe(kfId);
        assert(kf);

        // Create camera pose vertex
        VertexPose* poseVertex = new VertexPose;
        poseVertex->setId(++vertexIndex);
        poseVertex->setEstimate(kf->GetTcw());
        poseVertex->setFixed(kf->GetId() == 0);
        optimizer_.addVertex(poseVertex);

        // Record in map
        kfIdToCovKfThenVertex_[kfId] = make_pair(kf, poseVertex);

        // Create mappoint vertices
        for (auto &[mptId, _] : kf->GetAllObservingMptIdToKptIdx())
        {
            if (mptIdToMptThenVertex_.count(mptId)) {
                continue;
            }

            auto mpt = mapManager_->GetMappoint(mptId);
            assert(mpt);
            if (mpt->GetObservedByKeyframeIds().size() == 1) {
                continue;
            }
            if (mpt->outlier_) {
                continue;
            }

            // Create mappoint vertex
            VertexMappoint* mptVertex = new VertexMappoint;
            mptVertex->setEstimate(mpt->GetPosition());
            mptVertex->setId(++vertexIndex);
            mptVertex->setMarginalized(true);
            optimizer_.addVertex(mptVertex);

            // Record in map
            mptIdToMptThenVertex_[mptId] = make_pair(mpt, mptVertex);
        }
    }

    int edgeIndex = 0;

    // Add all measurement edges, and fixed pose pose vertices, also perform triangulation
    size_t triangulatedCnt = 0;
    for (auto& [mptId, mptAndVertex] : mptIdToMptThenVertex_)
    {   
        auto& [mpt, mptVertex] = mptAndVertex;

        vector<SE3> poses;
        vector<Vector3d> normalizedPos;
        // TODO: enable triangulation
        // bool needTriangulate = !mpt->outlier_ && !(mpt->triangulated_ || mpt->optimized_);
        bool needTriangulate = false;

        for (auto &kfId : mpt->GetObservedByKeyframeIds()) {
            auto keyframe = mapManager_->GetKeyframe(kfId);
            assert(keyframe);

            const auto& optkptIdx = keyframe->GetMatchedKeypointIdxForMappoint(mptId);
            assert(optkptIdx.has_value());
            auto& measurement = keyframe->GetKeypoint(optkptIdx.value()).pt;

            // TODO: check is keyframe is outlier

            VertexPose* poseVertex;
            if (kfIdToCovKfThenVertex_.count(kfId)) {
                // If the keyframe is a covisible keyframe
                poseVertex = kfIdToCovKfThenVertex_[kfId].second;
            }
            else {
                // Otherwise this keyframe is a 2nd-order covisible keyframe.
                // It needs to be added into the graph optimization but it's pose should keep fixed.
                poseVertex = new VertexPose;
                poseVertex->setId(++vertexIndex);
                poseVertex->setEstimate(keyframe->GetTcw());
                poseVertex->setFixed(true);
                optimizer_.addVertex(poseVertex);

                // Record in map
                kfIdToFixedKfThenVertex_[kfId] = make_pair(keyframe, poseVertex);
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);

            edge->setVertex(0, poseVertex);
            edge->setVertex(1, mptVertex);
            edge->setId(++edgeIndex);
            edge->setMeasurement(toVector2d(measurement));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(sqrt(config_.baInlierThres));
            edge->setRobustKernel(rk);
            optimizer_.addEdge(edge);

            edges_.push_back({edge, false, keyframe, mpt});

            if (needTriangulate) {
                poses.push_back(keyframe->GetTcw());
                normalizedPos.push_back(keyframe->camera_->Pixel2Camera(measurement));
            }
        }

        if (needTriangulate) {
            Vector3d pworld = Vector3d::Zero();
            if (Triangulation(poses, normalizedPos, pworld) && pworld[2] > 0)
            {
                // if triangulate successfully
                mptVertex->setEstimate(pworld);
                mpt->triangulated_ = true;
                ++triangulatedCnt;
            }
        }
    }

    double adjustedInlierThres = config_.baInlierThres;
    size_t outlierCnt = 0;

    // Optimize 4 * 20 steps
    for(size_t iteration = 0; iteration < 4; ++iteration) {
        optimizer_.initializeOptimization(0);
        optimizer_.optimize(20);

        outlierCnt = 0;
        observingMptToRemove_.clear();

        for (auto& [edge, isOutlier, kf, mpt] : edges_) {
            // Compute error for outlier edges since they won't be calculated during optimization
            if (isOutlier) {
                edge->computeError();
            }

            // chi2 is the (u^2 + v^2)
            if (edge->chi2() > config_.baInlierThres) {
                // level 1 edges won't be optimized later
                edge->setLevel(1);
                isOutlier = true;
                observingMptToRemove_.push_back(make_pair(kf, mpt->GetId()));
                ++outlierCnt;
            } else {
                edge->setLevel(0);
                isOutlier = false;
            }

            mpt->optimized_ = isOutlier;
        }

        // Loose the threshold if the outlier is too much
        double outlierRatio = outlierCnt / double(edges_.size());
        if (outlierRatio > 0.5) {
            adjustedInlierThres *= 2;
        }
    }

    printf("[Backend] optimization results:\n");
    printf("  optimized pose count: %zu\n", kfIdToCovKfThenVertex_.size());
    printf("  fixed pose count: %zu\n", kfIdToFixedKfThenVertex_.size());
    printf("  optimized mappoint count: %zu\n", mptIdToMptThenVertex_.size());
    printf("  triangulated mappoints count: %zu\n", triangulatedCnt);
    printf("  edge count: %zu\n", edges_.size());
    printf("  outlier edge count: %zu\n\n", outlierCnt);
}

void Backend::UpdateFrontendTrackingMap() {

    // Also write update back at this step
    frontendMapUpdateHandler_([&](unordered_map<size_t, Mappoint::Ptr>& trackingMap){
        
        for(const auto& [kf, mptId]: observingMptToRemove_) {
            kf->RemoveObservingMappointCreatedFromOtherFrame(mptId);

            // mpt's observedBy keyframe changes, need to recalculate descriptor
            auto mpt = mapManager_->GetMappoint(mptId);
            mpt->UpdateDescriptor();
        }

        for (const auto &[_, kfAndVertex] : kfIdToCovKfThenVertex_) {
            auto& [kf, kfVertex] = kfAndVertex;
            const SE3 oldTcw = kf->GetTcw();
            const SE3 newTcw = kfVertex->estimate();
            kf->SetTcw(newTcw);

            const SE3 T = newTcw.inverse() * oldTcw;
            // Update the mappoint position only observed by this keyframe
            for(auto& mptId: kf->GetMappointIdsOnlyObservedByThisFrame()) {
                auto mpt = mapManager_->GetMappoint(mptId);
                assert(mpt);

                const Vector3d newPosition = T * mpt->GetPosition();
                mpt->SetPosition(newPosition);
            }
        }

        for (const auto &[mptId, mptAndVertex] : mptIdToMptThenVertex_) {
            auto& [mpt, mptVertex] = mptAndVertex;
            if (mpt->outlier_) {
                continue;
            }

            mpt->SetPosition(mptVertex->estimate());
            // since the mpt position and keyframe pose changes, update its norm direction
            mpt->UpdateNormViewDirection();
        }

        trackingMap.clear();
        // get more mappoints from all covisible keyframes of current keyframe
        mapManager_->GetMappointsNearKeyframe(keyframeCurr_, trackingMap);
    });
}

void Backend::CleanUp() {

    kfIdToCovKfThenVertex_.clear();
    mptIdToMptThenVertex_.clear();
    kfIdToFixedKfThenVertex_.clear();
    edges_.clear();

    observingMptToRemove_.clear();

    // The algorithm, vertex and edges will be deallocated by g2o
    optimizer_.clear();
}

} // namespace