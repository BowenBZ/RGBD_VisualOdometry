#include "myslam/backend.h"

#include "myslam/util.h"
#include "myslam/mapmanager.h"

namespace myslam
{

Backend::Backend(const Camera::Ptr camera): camera_(move(camera)) {
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<CSparseLinearSolverType>()));
    optimizer_.setAlgorithm(solver);
    
    chi2Threshold_ = Config::get<double>("chi2_th");
    reMatchDescriptorDistance_ = Config::get<double>("re_match_descriptor_distance");

    backendRunning_ = true;
    backendThread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::Stop() {
    backendRunning_ = false;
    backendUpdateTrigger_.notify_one();
    backendThread_.join();

    CleanUp();
}

void Backend::AddNewKeyframeInfo(const FrontendToBackendInfo& info) {
    unique_lock<mutex> lock(backendMutex_);

    FrontendToBackendInfo copyInfo;
    copyInfo.keyframe = info.keyframe;
    copyInfo.oldMptIdKptIdxMap.insert(info.oldMptIdKptIdxMap.begin(), info.oldMptIdKptIdxMap.end());
    copyInfo.newMptKptIdxMap.insert(info.newMptKptIdxMap.begin(), info.newMptKptIdxMap.end());

    frontendInfoToProcess_.push(copyInfo);
    backendUpdateTrigger_.notify_one();
}

void Backend::BackendLoop()
{
    while (backendRunning_)
    {
        unique_lock<mutex> lock(backendMutex_);
        if (frontendInfoToProcess_.size() == 0) {
            backendUpdateTrigger_.wait(lock);
        }

        // Need to check again because the trigger could also be triggered during deconstruction
        if (backendRunning_)
        {
            printf("[Backend] info queue size: %zu\n", frontendInfoToProcess_.size());
            PopInfoFromQueue();
            printf("[Backend] starts processing new frame: %zu\n", keyframeCurr_->GetId());
            // unlock so that frontend could keep sending info
            lock.unlock();

            // ProjectMoreMappointsToNewKeyframe();
            MapManager::Instance().AddKeyframe(keyframeCurr_);
            AddObservingMappointsToNewKeyframe();
            AddNewMappointsToExistingKeyframe();
            OptimizeLocalMap();
            UpdateFrontendTrackingMap();
            CleanUp();
        }
    }
}

void Backend::PopInfoFromQueue() {

    auto& info = frontendInfoToProcess_.front();

    keyframePrev_ = keyframeCurr_;
    keyframeCurr_ = info.keyframe;
    oldMptIdKptIdxMap_.clear();
    oldMptIdKptIdxMap_.insert(info.oldMptIdKptIdxMap.begin(), info.oldMptIdKptIdxMap.end());
    newMptKptIdxMap_.clear();
    newMptKptIdxMap_.insert(info.newMptKptIdxMap.begin(), info.newMptKptIdxMap.end());

    frontendInfoToProcess_.pop();
}

void Backend::ProjectMoreMappointsToNewKeyframe() {
    if (keyframePrev_ == nullptr) {
        return;
    }

    unordered_map<size_t, Mappoint::Ptr> kptIdxNewMptMap;
    for(auto& [mpt, kptIdx]: newMptKptIdxMap_) {
        kptIdxNewMptMap[kptIdx] = mpt;
    }

    unordered_map<size_t, pair<size_t, double>> kptIdxToMptIdAndDistance;
    unordered_map<size_t, Mappoint::Ptr> oldMptIdToMpt;
    MapManager::Instance().GetMappointsNearKeyframe(keyframePrev_, oldMptIdToMpt);
    for(auto& [oldMptId, oldMpt]: oldMptIdToMpt) {
        // If this mpt already match with new keyframe
        if (oldMptIdKptIdxMap_.count(oldMptId)) {
            continue;
        }

        double distance;
        size_t kptIdx;
        bool mayObserveMpt;
        // Cannot find match
        if (!keyframeCurr_->GetMatchedKeypoint(oldMpt, false, kptIdx, distance, mayObserveMpt) || distance > reMatchDescriptorDistance_) {
            continue;
        }

        // this kpt already has matched previous mappoint
        if (!kptIdxNewMptMap.count(kptIdx)) {
            continue;
        }

        // there is other old mpt matched with this new kpt
        if (kptIdxToMptIdAndDistance.count(kptIdx) && kptIdxToMptIdAndDistance[kptIdx].second <= distance) {
            continue;
        }

        kptIdxToMptIdAndDistance[kptIdx] = make_pair(oldMptId, distance);
    }

    for(auto& [kptIdx, oldMptIdAndDistance]: kptIdxToMptIdAndDistance) {
        auto& [oldMptId, _] = oldMptIdAndDistance;
        oldMptIdKptIdxMap_[oldMptId] = kptIdx;
        auto newMpt = kptIdxNewMptMap[kptIdx];
        newMptKptIdxMap_.erase(newMpt);
    }

    printf("[Backend] Projected %zu old mpts to new keyframe\n", kptIdxToMptIdAndDistance.size());
}

void Backend::AddObservingMappointsToNewKeyframe() {
    // no need to update descriptor here since all mpts observed by current keyframe will be updated in the end of backend
    for (const auto& [mptId, kptIdx] : oldMptIdKptIdxMap_) {
        // old mpt may be replaced by previous new mpt
        auto mpt = MapManager::Instance().GetPotentialReplacedMappoint(mptId);
        // the mpt may already be considered as outlier
        if (mpt->outlier_) {
            continue;
        }
        keyframeCurr_->AddObservingMappoint(mpt, kptIdx);
    }

    // add new created mpts observations
    for (const auto& [mpt, kptIdx]: newMptKptIdxMap_) {
        MapManager::Instance().AddMappoint(mpt);
        keyframeCurr_->AddObservingMappoint(mpt, kptIdx);
    }
}

void Backend::AddNewMappointsToExistingKeyframe() {
    list<size_t> allCovisibleKfIds;
    keyframeCurr_->GetAllCovisibleKfIds(allCovisibleKfIds);
    unordered_set<Frame::Ptr> covisibleKfs;
    for (auto& kfId: allCovisibleKfIds) {
        auto kf = MapManager::Instance().GetKeyframe(kfId);
        if (kf == nullptr) {
            continue;
        }
        covisibleKfs.insert(kf);
        list<size_t> neighborAllCovisibleKfIds;
        kf->GetAllCovisibleKfIds(neighborAllCovisibleKfIds);
        for (auto& neighborKfId: neighborAllCovisibleKfIds) {
            auto neighborKf = MapManager::Instance().GetKeyframe(kfId);
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
    size_t oldMptId;

    for (auto& kf: covisibleKfs) {

        kptIdxToMptAndDistance.clear();
        for (auto& [mpt, _]: newMptKptIdxMap_) {
            if (!kf->GetMatchedKeypoint(mpt, false, kptIdx, distance, mayObserveMpt) || distance > reMatchDescriptorDistance_) {
                continue;
            }

            // check if the old keyframe keypoint already has matched mappoint
            if (kf->IsKeypointMatchWithMappoint(kptIdx, oldMptId)) {
                // if a previous matched keypoint could be matched with several new mappoints, find the best one for it
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
        kf->AddObservingMappoint(mpt, kptIdx);
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
        auto kf = MapManager::Instance().GetKeyframe(kfId);

        if (kf == nullptr)
        {
            continue;
        }

        // Create camera pose vertex
        VertexPose* poseVertex = new VertexPose;
        poseVertex->setId(++vertexIndex);
        poseVertex->setEstimate(kf->GetTcw());
        poseVertex->setFixed(kf->GetId() == 0);
        optimizer_.addVertex(poseVertex);

        // Record in map
        kfIdToCovKfThenVertex_[kfId] = make_pair(kf, poseVertex);

        // Create mappoint vertices
        list<size_t> observingMptIds;
        kf->GetObservingMappointIds(observingMptIds);
        for (auto &mptId : observingMptIds)
        {
            if (mptIdToMptThenVertex_.count(mptId))
            {
                continue;
            }

            auto mpt = MapManager::Instance().GetMappoint(mptId);
            if (mpt == nullptr || mpt->outlier_)
            {
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

    const double deltaRGBD = sqrt(7.815);
    int edgeIndex = 0;

    // Create pose vertices for fixed keyframe and add all edges, also perform triangulation
    size_t triangulatedCnt = 0;
    for (auto& [mptId, mptAndVertex] : mptIdToMptThenVertex_)
    {   
        auto& [mpt, mptVertex] = mptAndVertex;

        vector<SE3> poses;
        vector<Vector3d> normalizedPos;
        // TODO: enable triangulation
        // bool needTriangulate = !mpt->outlier_ && !(mpt->triangulated_ || mpt->optimized_);
        bool needTriangulate = false;

        unordered_map<size_t, size_t> observedByKfIdToKptIdx;
        mpt->GetObservedByKeyframesMap(observedByKfIdToKptIdx);
        for (auto &[kfId, kptIdx] : observedByKfIdToKptIdx)
        {
            auto keyframe = MapManager::Instance().GetKeyframe(kfId);
            auto& kpt = keyframe->GetKeypoint(kptIdx);

            if (keyframe == nullptr)
            {
                continue;
            }
            // TODO: check is keyframe is outlier

            VertexPose* poseVertex;
            // If the keyframe is covisible keyFrame
            if (kfIdToCovKfThenVertex_.count(kfId))
            {
                poseVertex = kfIdToCovKfThenVertex_[kfId].second;
            }
            else
            {
                // else needs to create a new vertex for fixed keyFrame
                VertexPose* fixedPoseVertex = new VertexPose;
                fixedPoseVertex->setId(++vertexIndex);
                fixedPoseVertex->setEstimate(keyframe->GetTcw());
                fixedPoseVertex->setFixed(true);
                optimizer_.addVertex(fixedPoseVertex);

                // Record in map
                kfIdToFixedKfThenVertex_[kfId] = make_pair(keyframe, fixedPoseVertex);

                poseVertex = fixedPoseVertex;
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);

            edge->setVertex(0, poseVertex);
            edge->setVertex(1, mptVertex);
            edge->setId(++edgeIndex);
            edge->setMeasurement(toVec2d(kpt.pt));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(deltaRGBD);
            edge->setRobustKernel(rk);
            optimizer_.addEdge(edge);

            edgeToKfThenMpt_[edge] = make_pair(keyframe, mpt);

            if (needTriangulate) {
                poses.push_back(keyframe->GetTcw());
                normalizedPos.push_back(keyframe->camera_->Pixel2Camera(kpt.pt));
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

    // Do first round optimization
    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // Find the outlier observations
    size_t outlierCnt = 0;
    for (auto& [edge, kfAndMpt] : edgeToKfThenMpt_)
    {
        edge->computeError();
        if (edge->chi2() > chi2Threshold_)
        {
            auto& [kf, mpt] = kfAndMpt;
            observingMptToRemove_.push_back(make_pair(kf, mpt->GetId()));
            edge->setLevel(1);
            ++outlierCnt;
        }
        edge->setRobustKernel(0);
    }

    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // Find more outlier observations
    for (auto& [edge, kfAndMpt] : edgeToKfThenMpt_)
    {
        // skip outlier of first round
        if (edge->level() != 0) {
            continue;
        }

        auto& [kf, mpt] = kfAndMpt;
        edge->computeError();
        if (edge->chi2() > chi2Threshold_) {
            observingMptToRemove_.push_back(make_pair(kf, mpt->GetId()));
            observingMptToRemoveSet_.insert(mpt);
            ++outlierCnt;
        } else {
            mpt->optimized_ = true;
        }
    }

    printf("[Backend] optimization results:\n");
    printf("  optimized pose count: %zu\n", kfIdToCovKfThenVertex_.size());
    printf("  fixed pose count: %zu\n", kfIdToFixedKfThenVertex_.size());
    printf("  optimized mappoint count: %zu\n", mptIdToMptThenVertex_.size());
    printf("  triangulated mappoints count: %zu\n", triangulatedCnt);
    printf("  edge count: %zu\n", edgeToKfThenMpt_.size());
    printf("  outlier edge count: %zu\n\n", outlierCnt);
}

void Backend::UpdateFrontendTrackingMap() {

    frontendMapUpdateHandler_([&](unordered_map<size_t, Mappoint::Ptr>& trackingMap){
        
        for (const auto &[_, kfAndVertex] : kfIdToCovKfThenVertex_) {
            auto& [kf, kfVertex] = kfAndVertex;
            kf->SetTcw(kfVertex->estimate());
        }

        for(const auto& [kf, mptId]: observingMptToRemove_) {
            kf->RemoveObservingMappoint(mptId);
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

        for(const auto& mpt: observingMptToRemoveSet_) {
            if (mpt->outlier_) {
                continue;
            }
            // mpt's observedBy keyframe changes, need to recalculate descriptor
            mpt->UpdateDescriptor();
        }

        trackingMap.clear();
        // get more mappoints from all covisible keyframes of current keyframe
        MapManager::Instance().GetMappointsNearKeyframe(keyframeCurr_, trackingMap);
    });
}

void Backend::CleanUp() {

    kfIdToCovKfThenVertex_.clear();
    mptIdToMptThenVertex_.clear();
    kfIdToFixedKfThenVertex_.clear();
    edgeToKfThenMpt_.clear();

    observingMptToRemove_.clear();
    observingMptToRemoveSet_.clear();

    // The algorithm, vertex and edges will be deallocated by g2o
    optimizer_.clear();
}

} // namespace