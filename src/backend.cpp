#include "myslam/backend.h"
#include "myslam/g2o_types.h"
#include "myslam/util.h"

namespace myslam {

void Backend::BackendLoop() {
    while (backendRunning_) {
        unique_lock<mutex> lock(backendMutex_);
        backendUpdateTrigger_.wait(lock);

        // Need to check again because the trigger could also be triggered during deconstruction
        if (backendRunning_) {
            Optimize();
        }
    }
}

void Backend::Optimize() {

    cout << "\nBackend starts optimization" << endl;

    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>>            idToCovisibleKeyframeThenVertex;
    unordered_map<size_t, pair<Mappoint::Ptr, VertexMappoint*>>     idToMappointThenVertex;
    // keyframes not belonging to covisible keyframes but could observe the local mappoints
    unordered_map<size_t, pair<Frame::Ptr, VertexPose*>>            idToFixedKeyframeThenVertex;
    
    unordered_map<BinaryEdgeProjection*, pair<Frame::Ptr, Mappoint::Ptr>>    edgeToKeyframeThenMappoint;

    auto covisibleKeyframesIds = keyframeCurr_->GetCovisibleKeyframes();
    // Add current keyframe 
    covisibleKeyframesIds.insert(keyframeCurr_->GetId());

    int vertexIndex = 0;

    // Create pose vertices and mappoint vertices for covisible keyframes 
    for(auto& keyframeId: covisibleKeyframesIds) {
    
        auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeId);

        if (keyframe == nullptr) {
            continue;
        }

        // Create camera pose vertex
        VertexPose *poseVertex = new VertexPose;
        poseVertex->setId(++vertexIndex);
        poseVertex->setEstimate(keyframe->GetPose());
        poseVertex->setFixed(keyframe->GetId() == 0);
        optimizer.addVertex(poseVertex);

        // Record in map
        idToCovisibleKeyframeThenVertex[keyframeId] = make_pair(keyframe, poseVertex);

        // Create mappoint vertices
        for(auto& mappointId: keyframe -> GetObservedMappointIds()) {
            if (idToMappointThenVertex.count(mappointId)) {
                continue;
            }

            auto mappoint = MapManager::GetInstance().GetMappoint(mappointId);
            if ( mappoint == nullptr || mappoint->outlier_ ) {
                continue;
            }

            // Create mappoint vertex
            VertexMappoint *mappointVertex = new VertexMappoint;
            mappointVertex->setEstimate(mappoint->GetPosition());
            mappointVertex->setId(++vertexIndex);
            mappointVertex->setMarginalized(true);
            optimizer.addVertex(mappointVertex);
            
            // Record in map
            idToMappointThenVertex[mappointId] = make_pair(mappoint, mappointVertex);
        }
    }

    const double deltaRGBD = sqrt(7.815);    
    int edgeIndex = 0;

    // Create pose vertices for fixed keyframe and add all edges 
    for (auto& pair : idToMappointThenVertex) {
        auto mappointId = pair.first;
        auto mappoint = pair.second.first;
        auto mappointVertex = pair.second.second;

        for(auto& observation: mappoint->GetObservedByKeyframesMap()) {
            auto keyframeId = observation.first;
            auto keyframe = MapManager::GetInstance().GetKeyframe(keyframeId);
            auto observedPixelPos = observation.second;

            if ( keyframe == nullptr ) {
                continue;
            }            
            // TODO: check is keyframe is outlier

            VertexPose* poseVertex;
            // If the keyframe is covisible keyFrame
            if ( idToCovisibleKeyframeThenVertex.count(keyframeId) ) {
                poseVertex = idToCovisibleKeyframeThenVertex[keyframeId].second;

            } else { 
                // else needs to create a new vertex for fixed keyFrame
                VertexPose* fixedPoseVertex = new VertexPose;
                fixedPoseVertex->setId(++vertexIndex);
                fixedPoseVertex->setEstimate(keyframe->GetPose());
                fixedPoseVertex->setFixed(true);
                optimizer.addVertex(fixedPoseVertex);

                // Record in map
                idToFixedKeyframeThenVertex[keyframeId] = make_pair(keyframe, fixedPoseVertex);

                poseVertex = fixedPoseVertex;
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);

            edge->setVertex(0, poseVertex);
            edge->setVertex(1, mappointVertex);
            edge->setId(++edgeIndex);
            edge->setMeasurement(toVec2d(observedPixelPos));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(deltaRGBD);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
            
            edgeToKeyframeThenMappoint[edge] = make_pair(keyframe, mappoint);
        }
    }

    // Do first round optimization
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Remove outlier and do second round optimization
    int outlierCnt = 0;
    for (auto &edgeToKeyframe : edgeToKeyframeThenMappoint) {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->chi2() > chi2Threshold_) {
            auto keyframe = edgeToKeyframeThenMappoint[edge].first;
            auto mappoint = edgeToKeyframeThenMappoint[edge].second;
            mappoint->RemoveObservedByKeyframe(keyframe->GetId());
            keyframe->RemoveObservedMappoint(mappoint->GetId());
            edge->setLevel(1);
            ++outlierCnt;
        } 
        edge->setRobustKernel(0);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Set outlier again
    for (auto &edgeToKeyframe : edgeToKeyframeThenMappoint) {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->level() == 0 && edge->chi2() > chi2Threshold_) {
            auto keyframe = edgeToKeyframeThenMappoint[edge].first;
            auto mappoint = edgeToKeyframeThenMappoint[edge].second;
            mappoint->RemoveObservedByKeyframe(keyframe->GetId());
            keyframe->RemoveObservedMappoint(mappoint->GetId());
            ++outlierCnt;
        } 
        edgeToKeyframeThenMappoint[edge].second->optimized_ = true;
    }

    cout << "\nBackend results:" << endl;
    cout << "  optimized pose number: "     << idToCovisibleKeyframeThenVertex.size() << endl;
    cout << "  optimized mappoint number: " << idToMappointThenVertex.size() << endl;
    cout << "  fixed pose number: "         << idToFixedKeyframeThenVertex.size() << endl;
    cout << "  edge number: "               << edgeToKeyframeThenMappoint.size() << endl;
    cout << "  outlier observations: "      << outlierCnt << endl;
    cout << endl;

    // Set pose and mappoint position
    for (auto& pair : idToCovisibleKeyframeThenVertex) {
        auto id = pair.first;
        auto keyframeAndVertex = pair.second;
        keyframeAndVertex.first -> SetPose(keyframeAndVertex.second->estimate());
    }
    for (auto& pair : idToMappointThenVertex) {
        auto id = pair.first;
        auto mappointAndVertex = pair.second;
        if (!mappointAndVertex.first->outlier_) {
            mappointAndVertex.first -> SetPosition(mappointAndVertex.second->estimate());
        }
    }
}




} // namespace 