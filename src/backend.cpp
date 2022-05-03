#include "myslam/backend.h"
#include "myslam/g2o_types.h"
#include "myslam/util.h"

namespace myslam {

void Backend::backendLoop() {
    while (backendRunning_) {
        unique_lock<mutex> lock(backendMutex_);
        mapUpdate_.wait(lock);

        // Need to check again because this will be trigger during the deconstruction
        if (backendRunning_) {
            optimize();
        }
    }
}

void Backend::optimize() {

    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // <Id, Frame::Ptr>
    unordered_map<uint64_t, Frame::Ptr> keyFrameMap;
    unordered_map<uint64_t, MapPoint::Ptr> mapPointMap;
    unordered_map<uint64_t, Frame::Ptr> fixedKeyFrameMap;
    
    // Optmized pose vertices
    unordered_map<uint64_t, VertexPose*> verticesPoseMap;
    // Mappoint vertices
    unordered_map<uint64_t, VertexMappoint*> verticesMappointMap;
    // Fixed pose vertices
    unordered_map<uint64_t, VertexPose*> fixedPoseVerticesMap;

    // Record edges
    unordered_map<BinaryEdgeProjection*, Frame::Ptr> edges_and_frames;
    unordered_map<BinaryEdgeProjection*, MapPoint::Ptr> edges_and_mappoint;

    int vertexIndex = 1;
    int edgeIndex = 1;

    auto connectedKeyFrames = keyFrameCurr_->getConnectedKeyFrames();

    // Add curr KeyFrame to the KeyFrame map
    connectedKeyFrames[keyFrameCurr_->getId()] = 0;

    // Find all keyFrames connected with current keyFrame
    for(auto& pair: connectedKeyFrames) {
    
        auto keyFrameId = pair.first;
        auto keyFrame = MapManager::GetInstance().GetKeyframe(keyFrameId);

        if (keyFrame == nullptr) {
            continue;
        }

        // Create camera pose vertex
        VertexPose *vertexPose = new VertexPose;
        vertexPose->setId(vertexIndex++);
        vertexPose->setEstimate(keyFrame->getPose());
        vertexPose->setFixed(keyFrame->getId() == 0);
        optimizer.addVertex(vertexPose);

        // Record in map
        keyFrameMap[keyFrame->getId()] = keyFrame;
        verticesPoseMap[keyFrame->getId()] = vertexPose;
    }

    // Find all mapppoints observed by keyFrameMap
    for (auto& pair : connectedKeyFrames) {

        auto keyFrameId = pair.first;
        auto keyFrame = MapManager::GetInstance().GetKeyframe(keyFrameId);
        
        if (keyFrame == nullptr) {
            continue;
        }

        for(auto& mapPointPtr: keyFrame -> getObservedMapPoints()) {
            if ( mapPointPtr.expired() ) {
                continue;
            }
            auto mapPoint = mapPointPtr.lock();
            if(mapPoint->outlier_) {
                continue;
            }

            // Create mapPoint vertex
            VertexMappoint *vertexMapPoint = new VertexMappoint;
            vertexMapPoint->setEstimate(mapPoint->getPosition());
            vertexMapPoint->setId(vertexIndex++);
            vertexMapPoint->setMarginalized(true);
            optimizer.addVertex(vertexMapPoint);
            
            // Record in map
            mapPointMap[mapPoint -> getId()] = mapPoint;
            verticesMappointMap[mapPoint->getId()] = vertexMapPoint;
        }
    }

    const double deltaRGBD = sqrt(7.815);    
    // Add all edges 
    for (auto& pair : mapPointMap) {
        auto mapPoint = pair.second;

        for(auto& observation: mapPoint->getKeyFrameObservationsMap()) {
            auto keyFrame = MapManager::GetInstance().GetKeyframe(observation.first);
            auto observedPixelPos = observation.second;

            if ( keyFrame == nullptr ) {
                continue;
            }            
            // TODO: check is keyframe is outlier

            VertexPose* edgePoseVertex;
            // If is connected keyFrame
            if ( keyFrameMap.count(keyFrame -> getId()) ) {
                edgePoseVertex = verticesPoseMap[keyFrame->getId()];
            } else { // else is fixed keyFrame
                fixedKeyFrameMap[keyFrame -> getId()] = keyFrame;

                // Create fixed pose vertex
                VertexPose* vertexPose = new VertexPose;
                vertexPose->setId(vertexIndex++);
                vertexPose->setEstimate(keyFrame->getPose());
                vertexPose->setFixed(true);
                optimizer.addVertex(vertexPose);

                // Record in map
                fixedPoseVerticesMap[keyFrame->getId()] = vertexPose;

                edgePoseVertex = vertexPose;
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);

            edge->setVertex(0, edgePoseVertex);
            edge->setVertex(1, verticesMappointMap[mapPoint->getId()]);
            edge->setId(edgeIndex++);
            edge->setMeasurement(toVec2d(observedPixelPos));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(deltaRGBD);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
            
            edges_and_frames[edge] = keyFrame;
            edges_and_mappoint[edge] = mapPoint;
        }
    }

    // Do first round optimization
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Remove outlier and do second round optimization
    int outlierCnt = 0;
    for (auto &ef : edges_and_frames) {
        auto edge = ef.first;
        edge->computeError();
        if (edge->chi2() > chi2_th_) {
            edges_and_mappoint[edge]->removeKeyFrameObservation(ef.second->getId());
            edges_and_frames[edge]->removeObservedMapPoint(edges_and_mappoint[edge]);
            edge->setLevel(1);
            outlierCnt++;
        } 
        edge->setRobustKernel(0);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Set outlier again
    for (auto &ef : edges_and_frames) {
        auto edge = ef.first;
        edge->computeError();
        if (edge->chi2() > chi2_th_ || edge->level() == 1) {
            edges_and_mappoint[edge]->removeKeyFrameObservation(ef.second->getId());
            edges_and_frames[edge]->removeObservedMapPoint(edges_and_mappoint[edge]);
            outlierCnt++;
        } 
        edges_and_mappoint[edge]->optimized_ = true;
    }

    cout << "\nBackend:" << endl;
    cout << "  optimized pose number: " << verticesPoseMap.size() << endl;
    cout << "  fixed pose number: " << fixedPoseVerticesMap.size() << endl;
    cout << "  mappoint/edge number: " << verticesMappointMap.size() << endl;
    cout << "  outlier observations: " << outlierCnt << endl;
    cout << endl;

    // Set pose and mappoint position
    for (auto& v : verticesPoseMap) {
        keyFrameMap[v.first]->setPose(v.second->estimate());
    }
    for (auto& v : verticesMappointMap) {
        if (!mapPointMap[v.first]->outlier_) {
            mapPointMap[v.first]->setPosition(v.second->estimate());
        }
    }
}




} // namespace 