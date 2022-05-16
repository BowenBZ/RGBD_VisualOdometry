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

    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    unordered_map<size_t, Frame::Ptr>       covisibleKeyframesMap;
    unordered_map<size_t, Mappoint::Ptr>    localMappointsMap;
    // keyframes not belonging to covisible keyframes but could observe the local mappoints
    unordered_map<size_t, Frame::Ptr>       fixedKeyframeMap;
    
    unordered_map<size_t, VertexPose*>      poseVerticesMap;
    unordered_map<size_t, VertexMappoint*>  mappointVerticesMap;
    unordered_map<size_t, VertexPose*>      fixedPoseVerticesMap;

    unordered_map<BinaryEdgeProjection*, Frame::Ptr>    edgesToKeyframes;
    unordered_map<BinaryEdgeProjection*, Mappoint::Ptr> edgesToMappoints;

    int vertexIndex = 0;

    auto covisibleKeyframesIdToWeight = keyframeCurr_->GetCovisibleKeyframes();
    // Add current keyframe 
    covisibleKeyframesIdToWeight[keyframeCurr_->GetId()] = 0;

    // Create pose vertices and mappoint vertices for covisible keyframes 
    for(auto& idToWeight: covisibleKeyframesIdToWeight) {
    
        auto keyframeId = idToWeight.first;
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
        covisibleKeyframesMap[keyframeId] = keyframe;
        poseVerticesMap[keyframeId] = poseVertex;

        // Create mappoint vertices
        for(auto& mappointId: keyframe -> GetObservedMappointIds()) {
            if (localMappointsMap.count(mappointId)) {
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
            localMappointsMap[mappointId] = mappoint;
            mappointVerticesMap[mappointId] = mappointVertex;
        }
    }

    const double deltaRGBD = sqrt(7.815);    
    int edgeIndex = 0;

    // Create pose vertices for fixed keyframe and add all edges 
    for (auto& pair : localMappointsMap) {
        auto mappointId = pair.first;
        auto mappoint = pair.second;

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
            if ( covisibleKeyframesMap.count(keyframeId) ) {
                poseVertex = poseVerticesMap[keyframeId];

            } else { 
                // else needs to create a new vertex for fixed keyFrame
                VertexPose* fixedPoseVertex = new VertexPose;
                fixedPoseVertex->setId(vertexIndex++);
                fixedPoseVertex->setEstimate(keyframe->GetPose());
                fixedPoseVertex->setFixed(true);
                optimizer.addVertex(fixedPoseVertex);

                // Record in map
                fixedKeyframeMap[keyframeId] = keyframe;
                fixedPoseVerticesMap[keyframeId] = fixedPoseVertex;

                poseVertex = fixedPoseVertex;
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);

            edge->setVertex(0, poseVertex);
            edge->setVertex(1, mappointVerticesMap[mappointId]);
            edge->setId(++edgeIndex);
            edge->setMeasurement(toVec2d(observedPixelPos));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(deltaRGBD);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
            
            edgesToKeyframes[edge] = keyframe;
            edgesToMappoints[edge] = mappoint;
        }
    }

    // Do first round optimization
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Remove outlier and do second round optimization
    int outlierCnt = 0;
    for (auto &edgeToKeyframe : edgesToKeyframes) {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->chi2() > chi2Threshold_) {
            edgesToMappoints[edge]->RemoveObservedByKeyframe(edgeToKeyframe.second->GetId());
            edgesToKeyframes[edge]->RemoveObservedMappoint(edgesToMappoints[edge]->GetId());
            edge->setLevel(1);
            ++outlierCnt;
        } 
        edge->setRobustKernel(0);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Set outlier again
    for (auto &edgeToKeyframe : edgesToKeyframes) {
        auto edge = edgeToKeyframe.first;
        edge->computeError();
        if (edge->level() == 0 && edge->chi2() > chi2Threshold_) {
            edgesToMappoints[edge]->RemoveObservedByKeyframe(edgeToKeyframe.second->GetId());
            edgesToKeyframes[edge]->RemoveObservedMappoint(edgesToMappoints[edge]->GetId());
            ++outlierCnt;
        } 
        edgesToMappoints[edge]->optimized_ = true;
    }

    cout << "\nBackend:" << endl;
    cout << "  optimized pose number: " << poseVerticesMap.size() << endl;
    cout << "  fixed pose number: " << fixedPoseVerticesMap.size() << endl;
    cout << "  mappoint/edge number: " << mappointVerticesMap.size() << endl;
    cout << "  outlier observations: " << outlierCnt << endl;
    cout << endl;

    // Set pose and mappoint position
    for (auto& v : poseVerticesMap) {
        covisibleKeyframesMap[v.first]->SetPose(v.second->estimate());
    }
    for (auto& v : mappointVerticesMap) {
        if (!localMappointsMap[v.first]->outlier_) {
            localMappointsMap[v.first]->SetPosition(v.second->estimate());
        }
    }
}




} // namespace 