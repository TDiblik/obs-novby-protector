#!/bin/bash

IS_LINUX=false
if [[ $(uname) == "Linux" ]]; then
    IS_LINUX=true
fi

if [[ "$IS_LINUX" == true ]]; then
    TEST_IMG_PATH="$(pwd)"
else
    TEST_IMG_PATH="$(pwd -W)"
fi
TEST_IMG_PATH="$TEST_IMG_PATH/img_test_sfw.jpg"

echo "----- Building -----";
cmake -B build/
cmake --build build/ --config Release

echo "----- Executing -----"
if [[ "$IS_LINUX" == true ]]; then
    cd ./build/
    ./NudeNetCPPDemo $TEST_IMG_PATH
    cd ..
else
    cd ./build/Release/
    ./NudeNetCPPDemo.exe $TEST_IMG_PATH
    cd ../../
fi