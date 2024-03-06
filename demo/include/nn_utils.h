#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <onnxruntime_c_api.h>
#include <opencv2/core/types.hpp>
#include "constants.h"

/*
   ----------------------------
   ---- IMAGE MANIPULATION ----
   ----------------------------
*/

/**
 * Scales a bounding box from the shape of the input image to the shape of an original image.
 *
 * @param img1_shape The shape (height, width) of the input image for the model.
 * @param box The bounding box to be scaled, specified as cv::Rect_<float>.
 * @param img0_shape The shape (height, width) of the original target image.
 * @param ratio_pad An optional parameter that specifies scaling and padding factors as a pair of values.
 *	The first value (ratio) is used for scaling, and the second value (pad) is used for padding.
 *	If not provided, default values will be used.
 * @param padding An optional boolean parameter that specifies whether padding should be applied.
 *	If set to true, padding will be applied to the bounding box.
 *
 * @return A scaled bounding box specified as cv::Rect_<float>.
 *
 * This function rescales a bounding box from the shape of the input image (img1_shape) to the shape of an original image (img0_shape).
 */
cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape, std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true);

void clip_boxes(cv::Rect& box, const cv::Size& shape);
void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape);
void clip_coords(std::vector<float>& coords, const cv::Size& shape);

std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape);

cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box);

// bboxes, confidences, classes, rest
std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>

non_max_suppression(const cv::Mat& output0, int class_names_num, int total_features_num, double conf_threshold, float iou_threshold);

void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    cv::Scalar_<double> color = cv::Scalar(), bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32
);

/*
   ----------------------------
   ----- HELPER FUNCTIONS -----
   ----------------------------
*/
#if TIMING_INFO
class Timer {
public:
    Timer(double& accumulator, bool isEnabled = true);
    void Stop();

private:
    double& accumulator;
    bool isEnabled;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
#endif

// Main purpose of this function is to parse `imgsz` key value of model metadata. Expected input: something like [544, 960] or [3,544, 960]
std::vector<int> parse_imgsz_from_metadata(const std::string& input);

// Main purpose of this function is to parse `names` key value of model metadata. Expected input: something like {Key: 0, Value: 'IDENTIFIER'}
std::unordered_map<int, std::string> parse_names_from_metadata(const std::string& input);

int64_t vector_product(const std::vector<int64_t>& vec);

#endif // NN_UTILS_H