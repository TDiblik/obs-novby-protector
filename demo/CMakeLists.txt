cmake_minimum_required(VERSION 3.5)
project(NudeNet_CPP_Demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(AIModelPath ${PROJECT_SOURCE_DIR}/../nudenet-best.onnx)
set(OpenCV_RELEASE_DLL_FILENAME opencv_world490.dll)

if (WIN32)
        set(OpenCV_DIR "C:/Program Files/opencv-4.9.0/build/x64/vc16/lib")      # opencv lib root
        set(OpenCV_BIN_DIR "C:/Program Files/opencv-4.9.0/build/x64/vc16/bin")  # opencv bin root
        set(ONNXRUNTIME_DIR "C:/Program Files/onnxruntime-win-x64-1.17.1")      # onnxruntime root
else ()
        set(OpenCV_DIR /usr/local/share/opencv-4.9.0/build/lib)      # opencv lib root
        set(OpenCV_BIN_DIR /usr/local/share/opencv-4.9.0/build/bin)  # opencv bin root
        set(ONNXRUNTIME_DIR /usr/local/share/onnxruntime-linux-x64-1.17.1/)  # onnxruntime root
endif ()

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
else ()
    target_link_libraries(NudeNetCPPDemo "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif ()

# copy the nudenet model
add_custom_command(TARGET NudeNetCPPDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AIModelPath}"
        "$<TARGET_FILE_DIR:NudeNetCPPDemo>"
        )