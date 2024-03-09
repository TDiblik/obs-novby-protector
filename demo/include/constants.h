#ifndef INCL_CONSTANTS_H
#define INCL_CONSTANTS_H

#include <string>
#include <opencv2/core.hpp>

#define ORT_VERBOSE false
#define DEBUG_INFO false
#define TIMING_INFO true

namespace MetadataConstants {
    inline const std::string IMGSZ = "imgsz";
    inline const std::string STRIDE = "stride";
    inline const std::string NC = "nc";
    inline const std::string CH = "ch";
    inline const std::string DATE = "date";
    inline const std::string VERSION = "version";
    inline const std::string TASK = "task";
    inline const std::string BATCH = "batch";
    inline const std::string NAMES = "names";
}

namespace OnnxProviders {
    inline const std::string CPU = "cpu";
    inline const std::string CUDA = "cuda";
}

namespace OnnxInitializers {
    inline const int UNINITIALIZED_STRIDE = -1;
    inline const int UNINITIALIZED_NC = -1;
}

namespace Utils {
    // Padding value when letterbox changes image size ratio
    inline const int DEFAULT_LETTERBOX_PAD_VALUE = 114;
    static const cv::Scalar COLOR_RED = cv::Scalar(0, 0, 255);
    static const cv::Scalar COLOR_BLACK = cv::Scalar(0, 0, 0);
    static const cv::Scalar_<double> LETTERBOX_COLOR = cv::Scalar(Utils::DEFAULT_LETTERBOX_PAD_VALUE, Utils::DEFAULT_LETTERBOX_PAD_VALUE, Utils::DEFAULT_LETTERBOX_PAD_VALUE);
}

#endif // INCL_CONSTANTS_H