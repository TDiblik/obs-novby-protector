#include "nn/autobackend.h"

#include <iostream>
#include <ostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "constants.h"
#include "nn_utils.h"

namespace fs = std::filesystem;

AutoBackendOnnx::AutoBackendOnnx(const char* modelPath, const char* logid, const char* provider)
    : OnnxModelBase(modelPath, logid, provider) {
    const std::unordered_map<std::string, std::string>& base_metadata = OnnxModelBase::getMetadata();

    auto imgsz_iterator = base_metadata.find(MetadataConstants::IMGSZ);
    if (imgsz_iterator != base_metadata.end()) {
        std::vector<int> imgsz = parse_imgsz_from_metadata(imgsz_iterator->second);
        if (imgsz_.empty()) {
            imgsz_ = imgsz;
        }
    }
    else {
        std::cerr << "Warning: Cannot get imgsz value from metadata" << std::endl;
    }

    auto stride_item = base_metadata.find(MetadataConstants::STRIDE);
    if (stride_item != base_metadata.end()) {
        int stide_int = std::stoi(stride_item->second);
        if (stride_ == OnnxInitializers::UNINITIALIZED_STRIDE) {
            stride_ = stide_int;
        }
    }
    else {
        std::cerr << "Warning: Cannot get stride value from metadata" << std::endl;
    }

    auto names_item = base_metadata.find(MetadataConstants::NAMES);
    if (names_item != base_metadata.end()) {
        std::unordered_map<int, std::string> names = parse_names_from_metadata(names_item->second);
#if DEBUG_INFO
        std::cout << "***Names from metadata***" << std::endl;
        for (const auto& pair : names) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
#endif
        if (names_.empty()) {
            names_ = names;
        }
    }
    else {
        std::cerr << "Warning: Cannot get names value from metadata" << std::endl;
    }

    // post init number of classes - you can do that only and only if names_ is not empty and nc was not initialized previously
    if (nc_ == OnnxInitializers::UNINITIALIZED_NC && !names_.empty()) {
        nc_ = names_.size();
    }
    else {
        std::cerr << "Warning: Cannot get nc value from metadata (probably names wasn't set)" << std::endl;
    }

    if (!imgsz_.empty() && inputTensorShape_.empty()) {
        inputTensorShape_ = { 1, ch_, getHeight(), getWidth() };
    }

    if (!imgsz_.empty()) {
        cvSize_ = cv::Size(getWidth(), getHeight());
    }

    // task init
    auto task_item = base_metadata.find(MetadataConstants::TASK);
    if (task_item != base_metadata.end()) {
        std::string task = std::string(task_item->second);
        if (task_.empty()) {
            task_ = task;
        }
    }
    else {
        std::cerr << "Warning: Cannot get task value from metadata" << std::endl;
    }
}

const std::vector<int>& AutoBackendOnnx::getImgsz() { return imgsz_; }
const int& AutoBackendOnnx::getHeight() { return imgsz_[0]; }
const int& AutoBackendOnnx::getWidth() { return imgsz_[1]; }
const int& AutoBackendOnnx::getStride() { return stride_; }
const int& AutoBackendOnnx::getCh() { return ch_; }
const int& AutoBackendOnnx::getNc() { return nc_; }
const std::unordered_map<int, std::string>& AutoBackendOnnx::getNames() { return names_; }
const cv::Size& AutoBackendOnnx::getCvSize() { return cvSize_; }
const std::vector<int64_t>& AutoBackendOnnx::getInputTensorShape() { return inputTensorShape_; }
const std::string& AutoBackendOnnx::getTask() { return task_; }

std::vector<YoloResults> AutoBackendOnnx::predict_once(cv::Mat& image, float& conf, float& iou, float& mask_threshold, int conversionCode) {

    // 1. preprocess
#if TIMING_INFO
    double preprocess_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    Timer preprocess_timer = Timer(preprocess_time, true);
#endif
    float* blob = nullptr;
    std::vector<Ort::Value> inputTensors;
    if (conversionCode >= 0) {
        cv::cvtColor(image, image, conversionCode);
    }
    std::vector<int64_t> inputTensorShape;
    cv::Mat preprocessed_img;
    cv::Size new_shape = cv::Size(getWidth(), getHeight());
    const bool& scaleFill = false;  // false
    const bool& auto_ = false; // true
    letterbox(image, preprocessed_img, new_shape, cv::Scalar(), auto_, scaleFill, true, getStride());
    _fill_blob(preprocessed_img, blob, inputTensorShape);
    int64_t inputTensorSize = vector_product(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
    ));

    // 2. inference
#if TIMING_INFO
    preprocess_timer.Stop();
    Timer inference_timer = Timer(inference_time, true);
#endif
    std::vector<Ort::Value> outputTensors = forward(inputTensors);
#if TIMING_INFO
    inference_timer.Stop();
    Timer postprocess_timer = Timer(postprocess_time, true);
#endif

    // 3. postprocess
    std::vector<YoloResults> results;
    std::unordered_map<int, std::string> names = this->getNames();
    int class_names_num = names.size();

    ImageInfo img_info = { image.size() };
    std::vector<int64_t> outputTensor0Shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* all_data0 = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)outputTensor0Shape[2], (int)outputTensor0Shape[1]), CV_32F, all_data0).t();  // [bs, features, preds_num]=>[bs, preds_num, features]
    _postprocess_detects(output0, img_info, results, class_names_num, conf, iou);

#if TIMING_INFO
    postprocess_timer.Stop();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Speed (" << (preprocess_time + inference_time + postprocess_time) * 1000.0 << "ms" << "): ";
    std::cout << (preprocess_time * 1000.0) << "ms preprocess, " << (inference_time * 1000.0) << "ms inference, ";
    std::cout << (postprocess_time * 1000.0) << "ms postprocess ; with img shape ";
    std::cout << "(1, " << image.channels() << ", " << preprocessed_img.rows << ", " << preprocessed_img.cols << ")" << std::endl;
#endif
#if DEBUG_INFO
    std::cout << "image: " << preprocessed_img.rows << "x" << preprocessed_img.cols << ", " << results.size() << " object(s), shape: (1, " << image.channels() << ", " << preprocessed_img.rows << ", " << preprocessed_img.cols << ")" << std::endl;
#endif

    return results;
}


void AutoBackendOnnx::_postprocess_detects(cv::Mat& output0, ImageInfo image_info, std::vector<YoloResults>& output,
    int& class_names_num, float& conf_threshold, float& iou_threshold)
{
    output.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;
    // 4 - your default number of rect parameters {x, y, w, h}
    int data_width = class_names_num + 4;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r)
    {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold)
        {
            masks.emplace_back(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back((float)max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);

            cv::Rect_ <float> bbox = cv::Rect_ <float>(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            cv::Rect_<float> scaled_bbox = scale_boxes(getCvSize(), bbox, image_info.raw_size);

            boxes.push_back(scaled_bbox);
        }
        pdata += data_width; // next pred
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
    for (int idx : nms_result)
    {
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, image_info.raw_size.width, image_info.raw_size.height);
        YoloResults result = { class_ids[idx] ,confidences[idx] ,boxes[idx] };
        output.push_back(result);
    }
}

void AutoBackendOnnx::_fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {
    cv::Mat floatImage;
    if (inputTensorShape.empty()) {
        inputTensorShape = getInputTensorShape();
    }
    image.convertTo(floatImage, CV_32FC3, 1.0f / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

