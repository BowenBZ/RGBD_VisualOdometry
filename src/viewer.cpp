#include "myslam/viewer.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>


namespace myslam {


void Viewer::SetCurrentFrame(
    const Frame::Ptr& current_frame, 
    const unordered_set<size_t>& matchedKptsIdx,
    const unordered_set<size_t>& inlierKptsIdx) {

    unique_lock<mutex> lck(viewer_data_mutex_);
    current_frame_ = current_frame;
    matchedKptsIdx_.clear();
    matchedKptsIdx_.insert(matchedKptsIdx.begin(), matchedKptsIdx.end());
    inlierKptsIdx_.clear();
    inlierKptsIdx_.insert(inlierKptsIdx.begin(), inlierKptsIdx.end());
}

void Viewer::UpdateDrawingObjects() {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    all_keyframes_ = MapManager::GetInstance().GetAllKeyframes();
    all_mappoints_ = MapManager::GetInstance().GetAllMappoints();
    active_mappoints_ = all_mappoints_;
}

void Viewer::ThreadLoop() {
    pangolin::CreateWindowAndBind("RGBD_VO", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float red[3] = {1.0, 0, 0};

    while (!pangolin::ShouldQuit() && viewer_running_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        unique_lock<mutex> lock(viewer_data_mutex_);
        if (current_frame_) {
            DrawFrame(current_frame_, red);
            // FollowCurrentFrame(vis_camera);

            cv::Mat img = PlotFrameImage();
            cv::imshow("image", img);
            cv::waitKey(1);
        }

        // DrawOtherKeyFrames();
        DrawMapPoints();

        pangolin::FinishFrame();
        usleep(5000);
    }
}

void Viewer::DrawOtherKeyFrames() {
    const float normalColor[3] = {0, 0, 1.0};

    for (auto& kf : all_keyframes_) {
        if(kf.first == current_frame_->GetId())
            continue;

        DrawFrame(kf.second, normalColor);
    }
}

void Viewer::DrawMapPoints() {
    const float activeColor[3] = {0.992, 0.702, 0.329};
    const float normalColor[3] = {0, 1.0, 0};
    const float outliderColor[3] = {1.0, 0, 0};

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto& mappoint : all_mappoints_) {
        if (active_mappoints_.count(mappoint.first)) {
            glColor3f(activeColor[0], activeColor[1], activeColor[2]);
        }
        else {
            glColor3f(normalColor[0], normalColor[1], normalColor[2]);
        }
        auto pos = mappoint.second->GetPosition();
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}


void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    SE3 Twc = frame->GetPose().inverse();
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat*)m.data());

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else {
        glColor3f(color[0], color[1], color[2]);
    }

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    SE3 Twc = current_frame_->GetPose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out = current_frame_->GetImage().clone();
    for(size_t idx = 0; idx < current_frame_->GetKeypointsSize(); ++idx) {
        cv::circle(img_out, current_frame_->GetKeypoint(idx).pt, 2, 
                   GetKeypointColor(idx), 2);
    }
    return img_out;
}

cv::Scalar Viewer::GetKeypointColor(size_t kptIdx) {
    // inlier points are green
    if(inlierKptsIdx_.count(kptIdx)) {
        return cv::Scalar(0, 255, 0);
    }

    // Matched but outlier points are blue
    if(matchedKptsIdx_.count(kptIdx)) {
        return cv::Scalar(255, 0, 0);
    }

    // Unmatched point is red
    return cv::Scalar(0, 0, 255);
}

    
} // namespace