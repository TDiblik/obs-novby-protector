cmake_minimum_required(VERSION 3.5)
project(NudeNet_CPP_Demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(AIModelPath ${PROJECT_SOURCE_DIR}/../nudenet-best.onnx)

set(OpenCV_DIR C:/opencv/build/x64/vc16/lib)  # opencv lib root
set(OpenCV_BIN_DIR C:/opencv/build/x64/vc16/bin)  #opencv bin root

set(OpenCV_DIR C:/opencv/build/x64/vc16/lib)  # opencv lib root
set(OpenCV_BIN_DIR C:/opencv/build/x64/vc16/bin)  #opencv bin root

set(OpenCV_DEBUG_DLL_FILENAME opencv_world490d.dll)  # change filenames
set(OpenCV_RELEASE_DLL_FILENAME opencv_world490.dll)  # change filenames

# SET (ONNXRUNTIME_DIR your/path/to/onnxruntime-win-x64-1.15.1)  # onnxruntime root
set(ONNXRUNTIME_DIR "C:/Program Files/onnxruntime")  # onnxruntime root

find_package(OpenCV REQUIRED)

# --- Configure your project files ---
include_directories(include) 
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)
add_executable(NudeNetCPPDemo ${CURR_SOURCES})
target_include_directories(NudeNetCPPDemo PRIVATE "${ONNXRUNTIME_DIR}/include")
target_compile_features(NudeNetCPPDemo PRIVATE cxx_std_17)
target_link_libraries(NudeNetCPPDemo ${OpenCV_LIBS})
if (WIN32)
    target_link_libraries(NudeNetCPPDemo "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    # copy onnxruntime dll
    add_custom_command(TARGET NudeNetCPPDemo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:NudeNetCPPDemo>"
            )
    # add opencv release dll
    add_custom_command(TARGET NudeNetCPPDemo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:NudeNetCPPDemo>"
            )
endif(WIN32)
if (UNIX)
    target_link_libraries(NudeNetCPPDemo "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

# copy the nudenet model
add_custom_command(TARGET NudeNetCPPDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AIModelPath}"
        "$<TARGET_FILE_DIR:NudeNetCPPDemo>"
        )