#include <random>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

#include "constants.h"
#include "nn_utils.h"

namespace fs = std::filesystem;

#if TIMING_INFO
void benchmark(uint number_of_frames, AutoBackendOnnx& model, cv::Mat img, float conf_threshold, float iou_threshold, float mask_threshold, int conversion_code, bool plot_fast = true) {
    std::cout
        << std::endl
        << "--------------------------------------------------------" << std::endl
        << "--------------- Benchmarking " << number_of_frames << " frames ---------------" << std::endl
        << "--------------------------------------------------------" << std::endl
        << std::endl;

    double time_for_completion = 0.0;
    Timer timer = Timer(time_for_completion, true);

    for (uint i = 0; i < number_of_frames; i++) {
        cv::Mat test_img = img.clone();
        std::vector<YoloResults> objs = model.predict_once(test_img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
        std::unordered_map<int, std::string> names = model.getNames();
        if (plot_fast) {
            plot_results_fast(test_img, objs);
        }
        else {
            plot_results_with_classifications(test_img, objs, names);
        }
    }

    timer.Stop();
    time_for_completion *= 1000; // convert to ms
    std::cout
        << std::endl
        << "--------------------------------------------------------" << std::endl
        << "------------------------ RESULTS -----------------------" << std::endl
        << "--------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout
        << std::endl
        << "It took " << time_for_completion << "ms (" << time_for_completion / 1000 << "s) to complete " << number_of_frames << " frames." << std::endl
        << "That's average of " << (time_for_completion / static_cast<double>(number_of_frames)) << "ms per frame." << std::endl
        << "(this includes the TIMING_INFO overhead)" << std::endl << std::endl;
}
#endif


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

    // benchmark(1000, model, img, conf_threshold, iou_threshold, mask_threshold, conversion_code, true);

    std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    std::unordered_map<int, std::string> names = model.getNames();
    // plot_results_fast(img, objs);
    plot_results_with_classifications(img, objs, names, false);
    cv::imshow("img", img);
    cv::waitKey();

    return 0;
}
