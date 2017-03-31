## GENERAL PROJECT SETTINGS
cmake_minimum_required(VERSION 3.5.1)

set(PROJECT_NAME GPUGenie)
set(PROJECT_API_VERSION 1)
set(PROJECT_MINOR_VERSION 0.0)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
project(${PROJECT_NAME})
enable_testing()
include(CheckCXXCompilerFlag)


## DEPENDENCIES
find_package(CUDA REQUIRED)
find_package(Boost 1.56.0 REQUIRED COMPONENTS serialization)


## BUILD OPTIONS
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Wno-unused-parameter") #-Werror
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()
if (NOT CONFIGURED_ONCE)
  set(CMAKE_C_FLAGS "${warnings}"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
  set(CMAKE_CXX_FLAGS "${warnings}"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()
#SET(CUDA_NVCC_FLAGS "-arch=sm_35" CACHE STRING "nvcc flags" FORCE)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52")
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -lineinfo")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
option(BUILD_SHARED_LIBS OFF)
option(USE_MPI "Build distributed GENIE" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})
link_directories(${PROJECT_BINARY_DIR}/lib ${Boost_LIBRARY_DIRS})

## OUTPUT OPTIONS
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

## CMAKE SUBDIRECTORIES
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(static)
add_subdirectory(doc)

