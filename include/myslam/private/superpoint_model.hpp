#ifndef SUPERPOINT_MODEL_H
#define SUPERPOINT_MODEL_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <iostream>

typedef struct {
    int x;
    int y;
    float confidence;
} CornerPoint;

class SuperPointModel {
public: 
  typedef std::shared_ptr<SuperPointModel> Ptr;

    SuperPointModel(std::string modelPath, float confidenceThresh, float distThresh);

    auto Initialized() const -> bool { return _initialized; }

    /*
    Input
      img - OpenCV cv::Mat grayscale float32 input image in range [0,1].
    Output
      corners - vector of corners
      desc - vector of unit unit normalized descriptors. (256, N)
    */
    void Process(const cv::Mat& image, std::vector<CornerPoint>& corners, cv::Mat& desc);

private:
    bool _initialized;
    torch::jit::script::Module _module;
    float _confidenceThresh;
    float _distThresh;

    cv::Size _resize = {320, 240};

    // Size of each output cell. Keep this fixed.
    int _cell = 8;

    // Remove points this close to the border.
    int _borderRemove = 4;

    /*
    """
    Run a faster approximate Non-Max-Suppression on tensor corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN tensor with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinity norm distance.
    Returns
      out_corners - 3xN torch tensor with surviving corners.
    """
    */
    void nms_fast(const torch::Tensor& in_corners, const int imageHeight, const int imageWidth, torch::Tensor& out_corners);
};

#endif // SUPERPOINT_MODEL_H