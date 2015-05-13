PROJECT(example_01)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)

INCLUDE(FindCUDA)
SET(CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_ROOT_DIR})

LINK_LIBRARIES(
  "winmm.lib"
  "psapi.lib"
  "shlwapi.lib"
)

INCLUDE_DIRECTORIES(
  "./"
  "../"
)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_20")

CUDA_ADD_EXECUTABLE(
  example_01
  main.cu
)

INSTALL(TARGETS example_01 DESTINATION ${example_01_SOURCE_DIR})