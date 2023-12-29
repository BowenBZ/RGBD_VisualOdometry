#include "myslam/backend.h"
#include "myslam/util.h"

namespace myslam
{

Backend::Backend(const Camera::Ptr camera): camera_(move(camera)) {
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<CSparseLinearSolverType>()));
    optimizer_.setAlgorithm(solver);
    
    chi2Threshold_ = Config::get<float>("chi2_th");

    backendRunning_ = true;
    backendThread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::Stop() {
    backendRunning_ = false;
    backendUpdateTrigger_.notify_one();
    backendThread_.join();

    CleanUp();
}

void Backend::BackendLoop()
{
    while (backendRunning_)
    {
        unique_lock<mutex> lock(backendMutex_);
        backendUpdateTrigger_.wait(lock);

        // Need to check again because the trigger could also be triggered during deconstruction
        if (backendRunning_)
        {
            Optimize();
            UpdateFrontendTrackingMap();
            CleanUp();
        }
    }
}

void Backend::Optimize()
{
    cout << "\nBackend starts optimization" << endl;

    auto covisibleKfIds = keyframeCurr_->GetCovisibleKeyframes();
    // Add current keyframe
    covisibleKfIds.insert(keyframeCurr_->GetId());

    int vertexIndex = 0;

    // Create pose vertices and mappoint vertices for covisible keyframes
    for (auto &kfId : covisibleKfIds)
    {
        auto kf = MapManager::GetInstance().GetKeyframe(kfId);

        if (kf == nullptr)
        {
            continue;
        }

        // Create camera pose vertex
        VertexPose* poseVertex = new VertexPose;
        poseVertex->setId(++vertexIndex);
        poseVertex->setEstimate(kf->GetPose());
        poseVertex->setFixed(kf->GetId() == 0);
        optimizer_.addVertex(poseVertex);

        // Record in map
        kfIdToCovKfThenVertex_[kfId] = make_pair(kf, poseVertex);

        // Create mappoint vertices
        for (auto &mptId : kf->GetObservedMappointIds())
        {
            if (mptIdToMptThenVertex_.count(mptId))
            {
                continue;
            }

            auto mpt = MapManager::GetInstance().GetMappoint(mptId);
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

    // Create pose vertices for fixed keyframe and add all edges
    for (auto& [mptId, mptAndVertex] : mptIdToMptThenVertex_)
    {   
        auto& mpt = mptAndVertex.first;
        auto& mptVertex = mptAndVertex.second;

        for (auto &[kfId, kptIdx] : mpt->GetObservedByKeyframesMap())
        {
            auto keyframe = MapManager::GetInstance().GetKeyframe(kfId);
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
                fixedPoseVertex->setEstimate(keyframe->GetPose());
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
        }
    }

    // Do first round optimization
    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // Remove outlier and do second round optimization
    int outlierCnt = 0;
    for (auto &edgeToKeyframe : edgeToKfThenMpt_)
    {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->chi2() > chi2Threshold_)
        {
            auto& keyframe = edgeToKfThenMpt_[edge].first;
            auto& mappoint = edgeToKfThenMpt_[edge].second;
            keyframe->RemoveObservedMappoint(mappoint->GetId());
            edge->setLevel(1);
            ++outlierCnt;
        }
        edge->setRobustKernel(0);
    }

    optimizer_.initializeOptimization(0);
    optimizer_.optimize(10);

    // Set outlier again
    for (auto &edgeToKeyframe : edgeToKfThenMpt_)
    {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->level() == 0 && edge->chi2() > chi2Threshold_)
        {
            auto& keyframe = edgeToKfThenMpt_[edge].first;
            auto& mappoint = edgeToKfThenMpt_[edge].second;
            keyframe->RemoveObservedMappoint(mappoint->GetId());
            ++outlierCnt;
        }
        edgeToKfThenMpt_[edge].second->optimized_ = true;
    }

    cout << "\nBackend results:" << endl;
    cout << "  optimized pose number: " << kfIdToCovKfThenVertex_.size() << endl;
    cout << "  optimized mappoint number: " << mptIdToMptThenVertex_.size() << endl;
    cout << "  fixed pose number: " << kfIdToFixedKfThenVertex_.size() << endl;
    cout << "  edge number: " << edgeToKfThenMpt_.size() << endl;
    cout << "  outlier observations: " << outlierCnt << endl;
    cout << endl;
}

void Backend::UpdateFrontendTrackingMap() {

    trackingMapUpdateHandler_(keyframeCurr_, [&](){
        for (auto &[_, kfAndVertex] : kfIdToCovKfThenVertex_)
        {
            auto& [kf, kfVertex] = kfAndVertex;
            kf->SetPose(kfVertex->estimate());
        }

        for (auto &[_, mptAndVertex] : mptIdToMptThenVertex_)
        {
            auto& [mpt, mptVertex] = mptAndVertex;
            if (!mpt->outlier_)
            {
                mpt->SetPosition(mptVertex->estimate());
            }
        }
    });
}

void Backend::CleanUp() {

    kfIdToCovKfThenVertex_.clear();
    mptIdToMptThenVertex_.clear();
    kfIdToFixedKfThenVertex_.clear();
    edgeToKfThenMpt_.clear();

    // The algorithm, vertex and edges will be deallocated by g2o
    optimizer_.clear();
}

} // namespace