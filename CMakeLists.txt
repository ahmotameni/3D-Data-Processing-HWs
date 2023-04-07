cmake_minimum_required(VERSION 3.24)
project(Git_repo)

set(CMAKE_CXX_STANDARD 14)

include_directories(3DP_lab_1/SGM)

add_executable(Git_repo
        3DP_lab_1/SGM/main.cpp
        3DP_lab_1/SGM/sgm.cpp
        3DP_lab_1/SGM/sgm.h)
