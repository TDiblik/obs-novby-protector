#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include "utils/common.h"

#include <string>
#include <sstream>
#include <codecvt>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <string>
#include <regex>
#include <vector>
#include <onnxruntime_c_api.h>

#if TIMING_INFO
Timer::Timer(double& accumulator, bool isEnabled)
    : accumulator(accumulator), isEnabled(isEnabled) {
    if (isEnabled) {
        start = std::chrono::high_resolution_clock::now();
    }
}

// Stop the timer and update the accumulator
void Timer::Stop() {
    if (isEnabled) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        accumulator += duration;
    }
}
#endif

const ORTCHAR_T* get_ort_path(const char* modelPath) {
#ifdef _WIN32
    return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(modelPath).c_str();
#else
    return modelPath;
#endif
}

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
