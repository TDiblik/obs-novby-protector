#include <random>

#include <filesystem>
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"

namespace fs = std::filesystem;

void plot_results(cv::Mat img, std::vector<YoloResults>& results, std::unordered_map<int, std::string>& names,
    const cv::Size& shape
) {
    cv::Mat mask = img.clone();
    auto raw_image_shape = img.size();

    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;

        // Draw bounding box
        rectangle(img, res.bbox, color, 2);

        // Try to get the class name corresponding to the given class_idx
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
            class_name = std::to_string(res.class_idx);
        }

        // Create label
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color, -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);
}



int main(int argc, char** argv) {
    const std::string& modelPath = "./nudenet-best.onnx";

    if (argc != 2) {
        std::cout << "Error: You have pass an image path as an argument" << std::endl;
        return 1;
    }
    std::string img_path = argv[1];
    if (!fs::exists(img_path)) {
        std::cout << "Error: Specified image path does not exist" << std::endl;
        return 1;
    }

    fs::path imageFilePath(img_path);
    const std::string& onnx_provider = OnnxProviders::CPU; // "cpu";
    const std::string& onnx_logid = "NudeNetCPPDemo_onnx_log";
    float mask_threshold = 0.5f;  // in python it's 0.5 and you can see that at ultralytics/utils/ops.process_mask line 705 (ultralytics.__version__ == .160)
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;  //  0.70f;
    int conversion_code = cv::COLOR_BGR2RGB;

    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    std::unordered_map<int, std::string> names = model.getNames();

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    plot_results(img, objs, names, img.size());
    cv::imshow("img", img);
    cv::waitKey();
    return 0;
}
