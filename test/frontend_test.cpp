#include "myslam/private/frame.hpp"
#include "myslam/private/mappoint.hpp"
#include <gtest/gtest.h>

#include <memory>
#include <myslam/config.hpp>
#include <myslam/private/superpoint_model.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

const std::string configPath = "//Users//bowen//Source//RGBD_VO//config//default.yaml";
const std::string modelPath = "//Users//bowen//Source//RGBD_VO//model//superpoint_converted.pt";
const std::string assetPath = "//Users//bowen//Source//RGBD_VO//test//asset";

const std::string imageName1 = "rgb//1305031102.175304.png";
const std::string imageName2 = "rgb//1305031102.211214.png";

const std::string depthName1 = "depth//1305031102.160407.png";
const std::string depthName2 = "depth//1305031102.226738.png";

bool visualize = true;

myslam::SuperPointModel::Ptr ConstructSuperPointModel() {
    myslam::SuperPointModel::Ptr model(new myslam::SuperPointModel(
        modelPath, 
        myslam::Config::get<double>("superpoint.confidenceThresh"),
        myslam::Config::get<double>("superpoint.distThresh")));
    EXPECT_TRUE(model->Initialized());

    return model;
}

// Test feature extraction 
TEST(FRONTEND_TEST, FeatureExtraction) {
    myslam::Config::setParameterFile(configPath);

    myslam::SuperPointModel::Ptr model = ConstructSuperPointModel();

    myslam::Camera::Ptr camera(new myslam::Camera());
    myslam::FrameConfig::Ptr frameConfig(new myslam::FrameConfig());

    cv::Mat image = cv::imread(assetPath + "//" + imageName1);
    myslam::Frame::Ptr frame = myslam::Frame::CreateFrame(
        frameConfig, 
        0, 
        camera, 
        image, 
        cv::Mat());

    frame->ExtractKeypointsAndDescriptorsWithSuperPointModel(model);

    const size_t keypointSize = frame->GetKeypointsSize();
    EXPECT_EQ(keypointSize, 256);

    if (!visualize) {
        return;
    }

    for (size_t i = 0; i < keypointSize; i++) {
        const auto& kpt = frame->GetKeypoint(i);
        cv::circle(image, kpt.pt, 2, {0, 255, 0}, 2);
    }

    cv::imshow("FeatureExtraction", image);
    cv::waitKey(0);
}

// Test feature matching with active search
TEST(FRONTEND_TEST, FeatureMatchActiveSearch) {
    myslam::Config::setParameterFile(configPath);

    myslam::SuperPointModel::Ptr model = ConstructSuperPointModel();

    myslam::Camera::Ptr camera(new myslam::Camera());
    myslam::FrameConfig::Ptr frameConfig(new myslam::FrameConfig());

    // Extract feature from image1
    cv::Mat image1 = cv::imread(assetPath + "//" + imageName1);
    myslam::Frame::Ptr frame1 = myslam::Frame::CreateFrame(
        frameConfig, 
        0, 
        camera, 
        image1, 
        cv::imread(assetPath + "//" + depthName1));

    frame1->SetTcw(SE3());
    frame1->ExtractKeypointsAndDescriptorsWithSuperPointModel(model);
    const size_t keypointSize1 = frame1->GetKeypointsSize();
    EXPECT_EQ(keypointSize1, 256);

    // Extract feature form image2
    cv::Mat image2 = cv::imread(assetPath + "//" + imageName2);
    myslam::Frame::Ptr frame2 = myslam::Frame::CreateFrame(
        frameConfig, 
        0, 
        camera, 
        image2, 
        cv::Mat());

    frame2->SetTcw(SE3());
    frame2->ExtractKeypointsAndDescriptorsWithSuperPointModel(model);
    const size_t keypointSize2 = frame2->GetKeypointsSize();
    EXPECT_EQ(keypointSize2, 228);

    // Create 3d points from image1
    std::vector<myslam::Mappoint::Ptr> mpts;
    std::unordered_map<size_t, size_t> mptIdToIdx;
    for (size_t i = 0; i < keypointSize1; i++) {
        const auto& kpt = frame1->GetKeypoint(i);
        const double depth = frame1->GetDepth(kpt);
        if (depth < 0) {
            continue;
        }

        Vector3d mptPos = camera->Pixel2World(kpt, frame1->GetTcw(), depth);

        myslam::Mappoint::Ptr mpt = myslam::Mappoint::CreateMappoint(
            mptPos, 
            frame1->GetDescriptor(i),
            true);
        mpts.push_back(mpt);
        mptIdToIdx[mpt->GetId()] = i;
    }
    EXPECT_EQ(mpts.size(), 231);

    // Try to find match from image2
    std::unordered_map<size_t, std::pair<myslam::Mappoint::Ptr, float>> kptToMpt;
    float distanceThresh = 0.7;
    for(const auto& mpt: mpts) {
        const auto& matchResult = frame2->SearchSuperpointKeypointMatchCandidate(mpt, distanceThresh, std::nullopt);
        
        if (!matchResult.has_value()) {
            continue;
        }

        const auto kptIdx = matchResult->keypointIdx;
        const auto distance = matchResult->distance;
        if ((kptToMpt.count(kptIdx) && distance < kptToMpt[kptIdx].second) ||
            !kptToMpt.count(kptIdx)) {
            kptToMpt[kptIdx] = {mpt, distance} ;
        }
    }
    EXPECT_EQ(kptToMpt.size(), 196);
    // printf("Match size: %zu\n", kptToMpt.size());

    if (!visualize) {
        return;
    }

    // Draw points to image1
    for (size_t i = 0; i < keypointSize1; i++) {
        const auto& kpt = frame1->GetKeypoint(i);
        cv::circle(image1, kpt.pt, 2, {0, 255, 0}, 2);
    }

    // Draw points to image2
    for (size_t i = 0; i < keypointSize2; i++) {
        const auto& kpt = frame2->GetKeypoint(i);
        cv::circle(image2, kpt.pt, 2, {0, 255, 0}, 2);
    }

    // Draw projected points to image2
    cv::Mat projectedOnImage2 = image2.clone();
    for (const auto& mpt: mpts) {
        const auto& projectedPoint = camera->World2Pixel(mpt->GetPosition(), SE3());
        const auto& projectedPt = cv::Point(projectedPoint.x(), projectedPoint.y());

        cv::circle(projectedOnImage2, projectedPt, 2, {255, 0, 0}, 2);
    }

    for (size_t i = 0; i < keypointSize2; i++) {
        if (!kptToMpt.count(i)) {
            continue;
        }

        const auto& kpt2 = frame2->GetKeypoint(i);
        const auto& [mpt, _] = kptToMpt[i];
        const auto& projectedPoint = camera->World2Pixel(mpt->GetPosition(), SE3());
        const auto& projectedPt = cv::Point(projectedPoint.x(), projectedPoint.y());

        cv::line(projectedOnImage2, kpt2.pt, projectedPt, {255, 0, 0});
    }
    cv::imshow("FeatureMatch - projected to image2", projectedOnImage2);

    // Show the combined image
    cv::Mat combined;
    cv::hconcat(image1, image2, combined);

    // Draw matching line
    for (size_t i = 0; i < keypointSize2; i++) {
        if (!kptToMpt.count(i)) {
            continue;
        }

        const auto& kpt2 = frame2->GetKeypoint(i);
        const auto& [mpt, _] = kptToMpt[i];
        const auto& kpt1 = frame1->GetKeypoint(mptIdToIdx[mpt->GetId()]);

        const cv::Point2f shiftedKpt2(kpt2.pt.x + image1.size().width, kpt2.pt.y);
        cv::line(combined, kpt1.pt, shiftedKpt2, {255, 0, 0});
    }

    cv::imshow("Feature matching", combined);
    cv::waitKey(0);
}