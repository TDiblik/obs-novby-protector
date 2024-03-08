#!/bin/bash

rm -rf build_x86_64
cmake --preset linux-x86_64
cd build_x86_64
ninja
pkill obs
sudo cp obs-novby-protector.so /usr/lib/obs-plugins/
