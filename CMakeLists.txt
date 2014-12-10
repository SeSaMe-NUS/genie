cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME GaLG)
set(PROJECT_API_VERSION 1)
set(PROJECT_MINOR_VERSION 0.0)

project(${PROJECT_NAME})

find_package(CUDA QUIET REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS ON)
set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(src)
add_subdirectory(test)
