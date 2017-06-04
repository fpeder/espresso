cmake_minimum_required(VERSION 2.8)
project(espresso)

find_package(BLAS REQUIRED)
find_package(CUDA)

option(CMAKE_EXPORT_COMPILE_COMMANDS "" 1)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_50)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/src/kernels")
include_directories("${CMAKE_SOURCE_DIR}/include/layers")

add_subdirectory("src")
add_subdirectory("test")
