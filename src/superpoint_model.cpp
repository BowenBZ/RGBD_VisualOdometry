#include "superpoint_model.hpp"

#include <torch/nn/functional.h>

namespace myslam {

SuperPointModel::SuperPointModel(std::string modelPath, float confidenceThresh, float distThresh) {
    _confidenceThresh = confidenceThresh;
    _distThresh = distThresh;
    _initialized = false;
    try {
        _module = torch::jit::load(modelPath);
        _module.eval();
        _initialized = true;
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
    }
}

void SuperPointModel::Process(const cv::Mat& image, std::vector<CornerPoint>& corners, cv::Mat& desc) {
    corners.clear();
    desc = cv::Mat();

    cv::Mat resizedImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2GRAY);
    // Resize input image from 640x480 to 320x240
    cv::resize(resizedImage, resizedImage, {320, 240}, 0, 0, cv::INTER_LINEAR);
    // Input image is in CV_8U, need to convert to float type
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // 1. Process the input
    assert(resizedImage.channels() == 1);
    assert(resizedImage.type() == CV_32F);
    int imageHeight = resizedImage.rows;
    int imageWidth = resizedImage.cols;
    // Create a tensor from the cv::Mat. OpenCV cv::Mat dimensions are [height, width, channels]
    torch::Tensor tensor_image = torch::from_blob(
        resizedImage.data,
        { imageHeight, imageWidth, resizedImage.channels() },
        torch::kFloat32
    );

    // Permute dimensions from HxWxC to CxHxW
    tensor_image = tensor_image.permute({2, 0, 1});

    // Add a batch dimension, so the input size is (1, 1, 240, 320)
    tensor_image = tensor_image.unsqueeze(0);

    // 2. Run the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);
    auto outputs = _module.forward(inputs).toTuple();

    // Output size is (1, 65, 30, 40)
    torch::Tensor semi = outputs->elements()[0].toTensor();
    // Output size is (1, 256, 30, 40)
    torch::Tensor coarse_desc = outputs->elements()[1].toTensor();

    // 3. Process the output
    semi = semi.squeeze();

    // 3.1 Softmax
    semi = torch::exp(semi);
    semi = semi / (semi.sum(0) + 0.00001);

    // Remove dustbin
    torch::Tensor nodust = semi.slice(0, 0, semi.size(0) - 1);
    nodust = nodust.permute({1, 2, 0});

    // 3.2 Reshape to get full resolution heatmap (240, 320)
    int Hc = int(imageHeight / _cell);
    int Wc = int(imageWidth / _cell);
    torch::Tensor heatmap = torch::reshape(nodust, {Hc, Wc, _cell,_cell});
    heatmap = heatmap.permute({0, 2, 1, 3});
    heatmap = torch::reshape(heatmap, {Hc * _cell, Wc * _cell});

    // 3.3 Select points with confidence higher than threshold
    auto selected = torch::where(heatmap >= _confidenceThresh);
    torch::Tensor ys = selected[0];
    torch::Tensor xs = selected[1];
    // No points found
    if (xs.size(0) == 0) {
        return;
    }

    // pts is (3, N)
    torch::Tensor pts = torch::zeros({3, xs.size(0)});
    pts.index_put_({0}, xs);
    pts.index_put_({1}, ys);
    pts.index_put_({2}, heatmap.masked_select(heatmap >= _confidenceThresh));

    // 3.4 Apply NMS.
    torch::Tensor selectedPts;
    nms_fast(pts, imageHeight, imageWidth, selectedPts);

    // Sort by confidence.
    torch::Tensor values = selectedPts.index({2});
    torch::Tensor inds = torch::argsort(values, 0, true);
    selectedPts = selectedPts.index_select(1, inds);

    // 3.5 Remove points along border
    // For border removal we only need the x (row 0) and y (row 1) coordinates.
    // Create boolean masks for x and y coordinates.
    auto pts_x = selectedPts.index({0});
    auto pts_y = selectedPts.index({1});
    auto toremoveW = torch::logical_or(pts_x < _borderRemove, pts_x >= (imageWidth - _borderRemove));
    auto toremoveH = torch::logical_or(pts_y < _borderRemove, pts_y >= (imageHeight - _borderRemove));
    auto toremove = torch::logical_or(toremoveW, toremoveH);
    auto keep_mask = torch::logical_not(toremove);
    torch::Tensor keepIndices = torch::nonzero(keep_mask).squeeze();
    selectedPts = selectedPts.index_select(1, keepIndices);
    if (selectedPts.size(1) == 0) {
        return;
    }

    // Construct output pts
    auto accessor = selectedPts.accessor<float, 2>();
    // Since the image are resized to half res, we need to convert it back for pixel positions
    const int scaler = 2;
    for (int i = 0; i < selectedPts.size(1); i++) {
        corners.push_back({(int)accessor[0][i] * 2, (int)accessor[1][i] * 2, accessor[2][i]});
    }

    // Construct output descriptors
    selectedPts = selectedPts.slice(0, 0, 2);

    // Normalize coordinates
    selectedPts.index({0}) = selectedPts.index({0}).div(static_cast<float>(imageWidth) / 2.0f) - 1.0f;
    selectedPts.index({1}) = selectedPts.index({1}).div(static_cast<float>(imageHeight) / 2.0f) - 1.0f;

    // Transpose from shape [2, N] to [N, 2] and ensure contiguous memory.
    selectedPts = selectedPts.transpose(0, 1).contiguous();
    selectedPts = selectedPts.view({1, 1, -1, 2});

    // Interpolate descriptors using grid_sample with bilinear interpolation, zero padding, and align_corners=True.
    auto gridOptions = torch::nn::functional::GridSampleFuncOptions()
                        .mode(torch::kBilinear)
                        .padding_mode(torch::kZeros)
                        .align_corners(true);
    torch::Tensor descTensor = torch::nn::functional::grid_sample(coarse_desc, selectedPts, gridOptions);

    // --- Step 3: Reshape and normalize descriptor ---
    // The output of grid_sample is of shape [1, 256, 1, num_points].
    int D = coarse_desc.size(1);
    assert(D == 256);
    // We want to reshape it to [256, num_points]
    descTensor = descTensor.view({D, -1});        // reshape to [256, num_points]

    // Normalize each descriptor (each column) with its L2 norm.
    // Compute the norm along dimension 0 (for each column) and keep dimension.
    torch::Tensor norms = descTensor.norm(2, /*dim=*/0, /*keepdim=*/true);
    descTensor = descTensor / norms;
    descTensor = descTensor.to(torch::kCPU).contiguous();           // ensure on CPU

    desc = cv::Mat(descTensor.size(0), descTensor.size(1), CV_32F, descTensor.data_ptr<float>()).clone();
}

void SuperPointModel::nms_fast(const torch::Tensor& in_corners, const int imageHeight, const int imageWidth, torch::Tensor& out_corners) {
    // Create grid and index holders (H x W int tensors).
    torch::Tensor grid = torch::zeros({imageHeight, imageWidth}, torch::kInt32);
    torch::Tensor inds = torch::zeros({imageHeight, imageWidth}, torch::kInt32);

    // Sort by descending confidence.
    // in_corners[2] is a 1D tensor of confidences.
    auto conf = in_corners.index({2});
    // argsort with descending=true gives indices that sort in descending order.
    torch::Tensor inds1 = torch::argsort(conf, /*dim=*/0, /*descending=*/true);
    
    // Permute in_corners by inds1: corners = in_corners[:, inds1]
    torch::Tensor corners = in_corners.index_select(1, inds1);
    
    // Round first two rows (x and y) and cast to int.
    torch::Tensor rcorners = corners.slice(0, 0, 2).round().to(torch::kInt32);

    // Handle edge cases.
    if (rcorners.size(1) == 0) {
        out_corners = torch::Tensor();
    }
    if (rcorners.size(1) == 1) {
        out_corners = torch::Tensor(rcorners);
    }

    // Fill grid and inds with corner indices.
    int N = rcorners.size(1);
    auto rcorners_accessor = rcorners.accessor<int32_t, 2>(); // shape [2, N]
    for (int i = 0; i < N; i++) {
        int x = rcorners_accessor[0][i];  // x coordinate
        int y = rcorners_accessor[1][i];  // y coordinate
        // Note: assumes x in [0, W-1] and y in [0, H-1]
        grid.index_put_({y, x}, torch::tensor(1, torch::kInt32));
        inds.index_put_({y, x}, torch::tensor(i, torch::kInt32));
    }

    // Pad grid by dist_thresh.
    int pad = _distThresh;
    grid = torch::constant_pad_nd(grid, {pad, pad, pad, pad}, 0);

    // Iterate over corners (in sorted order) and suppress neighbors.
    for (int i = 0; i < N; i++) {
        int x = rcorners_accessor[0][i];
        int y = rcorners_accessor[1][i];
        int pt_x = x + pad;
        int pt_y = y + pad;
        // If this point is still active (==1) in the grid.
        if (grid.index({pt_y, pt_x}).item<int>() == 1) {
            // Set neighborhood to 0.
            int start_y = pt_y - pad;
            int end_y = pt_y + pad + 1;
            int start_x = pt_x - pad;
            int end_x = pt_x + pad + 1;
            grid.slice(0, start_y, end_y)
                .slice(1, start_x, end_x)
                .fill_(0);
            // Mark the center as kept (-1).
            grid.index_put_({pt_y, pt_x}, torch::tensor(-1, torch::kInt32));
        }
    }

    // Get surviving points: indices where grid == -1.
    torch::Tensor keep = torch::nonzero(grid == -1); // shape: [K, 2] with (y,x) coordinates
    // Subtract pad from coordinates.
    torch::Tensor keep_y = keep.select(1, 0) - pad;
    torch::Tensor keep_x = keep.select(1, 1) - pad;
    
    // Use advanced indexing to get indices from the original inds tensor.
    torch::Tensor inds_keep = inds.index({keep_y, keep_x});
    
    // Get surviving corners.
    out_corners = corners.index_select(1, inds_keep);
}

} // namespace myslam