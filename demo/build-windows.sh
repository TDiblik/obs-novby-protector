#!/bin/bash
echo "----- Building -----" && \
cmake -B build/ -DCMAKE_BUILD_TYPE=Release && \
cmake --build build/ --target ALL_BUILD --config Release && \ 
echo "----- Executing -----" && \
cd ./build/Release/ && \
./NudeNetCPPDemo.exe && \
cd ../../