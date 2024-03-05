#!/bin/bash
TEST_IMG_PATH="$(pwd -W)/img_test_sfw.jpg" && \
echo "----- Building -----" && \
cmake -B build/ && \
cmake --build build/ --target ALL_BUILD --config Release && \ 
echo "----- Executing -----" && \
cd ./build/Release/ && \
./NudeNetCPPDemo.exe $TEST_IMG_PATH && \
cd ../../