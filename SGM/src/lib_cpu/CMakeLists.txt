cmake_minimum_required(VERSION 2.8)
project(fp_sgbm)

set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -g -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(FPSTEREO_DIR "#PATH_TO_FP_STEREO#/fpStereo")

message(STATUS "opencv include dir:" ${OpenCV_INCLUDE_DIRS})

include_directories(
    ${PROJECT_SOURCE_DIR}
    "/tools/Xilinx/Vivado/2018.3/include"
    ${FPSTEREO_DIR}
)
link_directories("/tools/Xilinx/Vivado/2018.3/lnx64/tools/opencv/opencv_gcc")
add_executable(test_fp_sgbm
    fp_sgbm_c.cpp
)
target_link_libraries(test_fp_sgbm
opencv_core
opencv_highgui
opencv_imgproc
opencv_features2d
opencv_calib3d
opencv_flann
z
jpeg
dl
rt
opencv_contrib
dw
)