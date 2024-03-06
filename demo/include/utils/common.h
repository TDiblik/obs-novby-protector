#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <onnxruntime_c_api.h>

#include "constants.h"

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

const ORTCHAR_T* get_ort_path(const char* modelPath);

// Main purpose of this function is to parse `imgsz` key value of model metadata. Expected input: something like [544, 960] or [3,544, 960]
std::vector<int> parse_imgsz_from_metadata(const std::string& input);
// Main purpose of this function is to parse `names` key value of model metadata. Expected input: something like {Key: 0, Value: 'IDENTIFIER'}
std::unordered_map<int, std::string> parse_names_from_metadata(const std::string& input);
int64_t vector_product(const std::vector<int64_t>& vec);

#endif // UTILS_COMMON_H