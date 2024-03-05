#ifndef UTILS_AUGMENT_H
#define UTILS_AUGMENT_H

#include <opencv2/core/types.hpp>

void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    cv::Scalar_<double> color = cv::Scalar(), bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32
);

#endif // UTILS_AUGMENT_H