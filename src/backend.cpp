#include "myslam/backend.h"
#include "myslam/g2o_types.h"
#include "myslam/algorithm.h"

namespace myslam {

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        unique_lock<mutex> lock(backend_mutex_);
        map_update_.wait(lock);

        auto active_kfs = map_->getAllKeyFrames();
        auto active_mappoints = map_->getActiveMappoints();
        Optimize(active_kfs, active_mappoints);
    }
}

void Backend::Optimize(Map::KeyframeDict &keyframes,
                       Map::MappointDict &mappoints) {

    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

   
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        
    }

    // Record pose vertices
    std::unordered_map<unsigned long, VertexPose *> verticesPoseMap;
    // Record mappoint vertices
    std::unordered_map<unsigned long, VertexMappoint *> verticesMappointMap;
    // Record edges
    std::unordered_map<BinaryEdgeProjection *, Frame::Ptr> edges_and_frames;
    std::unordered_map<BinaryEdgeProjection *, MapPoint::Ptr> edges_and_mappoint;
    int vertex_index = 1;
    int edge_index = 1;
    double chi2_th = 5.991;

    for (auto &mappoint : mappoints) {
        unsigned long mappoint_id = mappoint.first;
        auto mp = mappoint.second;

        if(mp->outlier_)
            continue;

        for(auto& obs : mappoint.second->getFrameObservations()) {
            // Check whether this observation is in current active keyframe
            auto frame = obs.first.lock();
            if (frame == nullptr || !keyframes.count(frame->getID()))
                continue;

            // Add the frame vertex if haven't
            if (!verticesPoseMap.count(frame->getID())) {
                VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
                vertex_pose->setId(vertex_index++);
                vertex_pose->setEstimate(frame->getPose());
                optimizer.addVertex(vertex_pose);

                // Record in map
                verticesPoseMap[frame->getID()] = vertex_pose;
            }
           
            // Add mappoint vertex if haven't
            if (!verticesMappointMap.count(mappoint_id)) {
                VertexMappoint *vertex_mappoint = new VertexMappoint;
                vertex_mappoint->setEstimate(mp->getPosition());
                vertex_mappoint->setId(vertex_index++);
                vertex_mappoint->setMarginalized(true);
                optimizer.addVertex(vertex_mappoint);

                // Record in map
                verticesMappointMap[mappoint_id] = vertex_mappoint;
            }

            // Add edge
            BinaryEdgeProjection* edge = new BinaryEdgeProjection(camera_);
            // Set two connecting vertex
            edge->setVertex(0, verticesPoseMap[frame->getID()]);    // pose vertex
            edge->setVertex(1, verticesMappointMap[mappoint_id]);        // mappoint vertex

            edge->setId(edge_index++);
            edge->setMeasurement(toVec2d(obs.second));
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
            
            edges_and_frames[edge] = frame;
            edges_and_mappoint[edge] = mp;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // remove outlier observations from mappoint
    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_frames) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }
    cnt_outlier = 0;
    for (auto &ef : edges_and_frames) {
        if (ef.first->chi2() > chi2_th) {
            edges_and_mappoint[ef.first]->removeFrameObservation(ef.second);
            cnt_outlier++;
        }
    }

    cout << "\nBackend:" << endl;
    cout << "  pose number: " << verticesPoseMap.size() << endl;
    cout << "  mappoint/edge number: " << verticesMappointMap.size() << endl;
    cout << "  outlier observations: " << cnt_outlier << endl;
    cout << endl;

    // Set pose and mappoint position
    for (auto &v : verticesPoseMap) {
        keyframes[v.first]->setPose(v.second->estimate());
    }
    for (auto &v : verticesMappointMap) {
        mappoints[v.first]->setPosition(v.second->estimate());
    }

}


} // namespace 