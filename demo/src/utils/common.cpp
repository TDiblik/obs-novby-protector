#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include <string>
#include <sstream>
#include <codecvt>
#include "utils/common.h"

#include <chrono>

#include <iostream>
#include <stdio.h>

#include <string>

#include <regex>
#include <vector>

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

// С++ 14 version
std::wstring get_win_path(const std::string& modelPath) {
#ifdef _WIN32
    return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(modelPath);
#else
    return std::wstring(modelPath.begin(), modelPath.end());
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
