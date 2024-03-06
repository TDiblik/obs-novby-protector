#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <regex>
#include <onnxruntime_c_api.h>

#include "constants.h"
#include "nn_utils.h"

/*
   ----------------------------
   ---- IMAGE MANIPULATION ----
   ----------------------------
*/

void clip_boxes(cv::Rect& box, const cv::Size& shape) {
    box.x = std::max(0, std::min(box.x, shape.width));
    box.y = std::max(0, std::min(box.y, shape.height));
    box.width = std::max(0, std::min(box.width, shape.width - box.x));
    box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}

void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape) {
    for (cv::Rect& box : boxes) {
        clip_boxes(box, shape);
    }
}

void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape) {
    for (cv::Rect_<float>& box : boxes) {
        clip_boxes(box, shape);
    }
}

// source: ultralytics/utils/ops.py scale_boxes lines 99+ (ultralytics==8.0.160)
cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape, std::pair<float, cv::Point2f> ratio_pad, bool padding) {
    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
            static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
        pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
        pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    cv::Rect_<float> scaledCoords(box);

    if (padding) {
        scaledCoords.x -= pad_x;
        scaledCoords.y -= pad_y;
    }

    scaledCoords.x /= gain;
    scaledCoords.y /= gain;
    scaledCoords.width /= gain;
    scaledCoords.height /= gain;

    // Clip the box to the bounds of the image
    clip_boxes(scaledCoords, img0_shape);

    return scaledCoords;
}

// Assuming coords are of shape [1, 17, 3]
void clip_coords(std::vector<float>& coords, const cv::Size& shape) {
    for (int i = 0; i < coords.size(); i += 3) {
        coords[i] = std::min(std::max(coords[i], 0.0f), static_cast<float>(shape.width - 1));  // x
        coords[i + 1] = std::min(std::max(coords[i + 1], 0.0f), static_cast<float>(shape.height - 1));  // y
    }
}

// source: ultralytics/utils/ops.py scale_coords lines 753+ (ultralytics==8.0.160)
std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape)
{
    std::vector<float> scaledCoords = coords;

    // Calculate gain and padding
    double gain = std::min(static_cast<double>(img1_shape.width) / img0_shape.width, static_cast<double>(img1_shape.height) / img0_shape.height);
    cv::Point2d pad((img1_shape.width - img0_shape.width * gain) / 2, (img1_shape.height - img0_shape.height * gain) / 2);

    // Apply padding. Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] -= pad.x;  // x padding
        scaledCoords[i + 1] -= pad.y;  // y padding
    }

    // Scale coordinates. Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] /= gain;
        scaledCoords[i + 1] /= gain;
    }

    clip_coords(scaledCoords, img0_shape);
    return coords;
}


cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box) {
    int h = mask.rows;
    int w = mask.cols;

    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.width;
    int y2 = box.y + box.height;

    cv::Mat cropped_mask = cv::Mat::zeros(h, w, mask.type());

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            if (r >= y1 && r < y2 && c >= x1 && c < x2) {
                cropped_mask.at<float>(r, c) = mask.at<float>(r, c);
            }
        }
    }

    return cropped_mask;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat& output0, int class_names_num, int data_width, double conf_threshold,
    float iou_threshold) {

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> rest;

    int rest_start_pos = class_names_num + 4;
    int rest_features = data_width - rest_start_pos;

    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold) {
            std::vector<float> mask_data(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back((float)max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
            cv::Rect_<float> bbox(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            boxes.push_back(bbox);
            if (rest_features > 0) {
                std::vector<float> rest_data(pdata + rest_start_pos, pdata + data_width);
                rest.push_back(rest_data);
            }
        }
        pdata += data_width; // next prediction
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result);
    std::vector<int> nms_class_ids;
    std::vector<float> nms_confidences;
    std::vector<cv::Rect> nms_boxes;
    std::vector<std::vector<float>> nms_rest;
    for (int idx : nms_result) {
        nms_class_ids.push_back(class_ids[idx]);
        nms_confidences.push_back(confidences[idx]);
        nms_boxes.push_back(boxes[idx]);
        nms_rest.push_back(rest[idx]);
    }
    return std::make_tuple(nms_boxes, nms_confidences, nms_class_ids, nms_rest);
}

void letterbox(
    const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape,
    cv::Scalar_<double> color,
    bool auto_,
    bool scaleFill,
    bool scaleUp,
    int stride
) {
    cv::Size shape = image.size();
    float r = std::min(
        static_cast<float>(newShape.height) / static_cast<float>(shape.height),
        static_cast<float>(newShape.width) / static_cast<float>(shape.width)
    );
    if (!scaleUp) {
        r = std::min(r, 1.0f);
    }

    float ratio[2]{ r, r };
    int newUnpad[2]{ static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                     static_cast<int>(std::round(static_cast<float>(shape.height) * r)) };

    auto dw = static_cast<float>(newShape.width - newUnpad[0]);
    auto dh = static_cast<float>(newShape.height - newUnpad[1]);
    if (auto_) {
        dw = static_cast<float>((static_cast<int>(dw) % stride));
        dh = static_cast<float>((static_cast<int>(dh) % stride));
    }
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / static_cast<float>(shape.width);
        ratio[1] = static_cast<float>(newShape.height) / static_cast<float>(shape.height);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    if (color == cv::Scalar()) {
        color = cv::Scalar(Utils::DEFAULT_LETTERBOX_PAD_VALUE, Utils::DEFAULT_LETTERBOX_PAD_VALUE, Utils::DEFAULT_LETTERBOX_PAD_VALUE);
    }

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


/*
   ----------------------------
   ----- HELPER FUNCTIONS -----
   ----------------------------
*/

#if TIMING_INFO
Timer::Timer(double& accumulator, bool isEnabled)
    : accumulator(accumulator), isEnabled(isEnabled) {
    if (isEnabled) {
        start = std::chrono::high_resolution_clock::now();
    }
}

void Timer::Stop() {
    if (isEnabled) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        accumulator += duration;
    }
}
#endif

std::vector<int> parse_imgsz_from_metadata(const std::string& input) {
    std::regex number_pattern(R"(\d+)");
    std::vector<std::string> strings;
    std::sregex_iterator it(input.begin(), input.end(), number_pattern);
    std::sregex_iterator end;

    while (it != end) {
        strings.push_back(it->str());
        ++it;
    }

    std::vector<int> result;
    for (const std::string& str : strings) {
        int value = std::stoi(str);
        result.push_back(value);
    }

    return result;
}

std::unordered_map<int, std::string> parse_names_from_metadata(const std::string& input) {
    std::unordered_map<int, std::string> result;

    std::string cleanedInput = input;
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '{'), cleanedInput.end());
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '}'), cleanedInput.end());

    std::istringstream elementStream(cleanedInput);
    std::string element;
    while (std::getline(elementStream, element, ',')) {
        std::istringstream keyValueStream(element);
        std::string keyStr, value;
        if (std::getline(keyValueStream, keyStr, ':') && std::getline(keyValueStream, value)) {
            int key = std::stoi(keyStr);
            result[key] = value;
        }
    }

    return result;
}

int64_t vector_product(const std::vector<int64_t>& vec) {
    int64_t result = 1;
    for (int64_t value : vec) {
        result *= value;
    }
    return result;
}