## PROJECT SETTINGS
cmake_minimum_required(VERSION 3.5.1)

set(PROJECT_NAME GPUGenie)
set(PROJECT_API_VERSION 1)
set(PROJECT_MINOR_VERSION 0.0)

project(${PROJECT_NAME})
enable_testing()
include(CheckCXXCompilerFlag)
include(ExternalProject)

option(COMPR "Build compression module" ON)
option(SIMDCAI "Build SIMDCAIR library (from Daniel Lemire) and related tests (COMPR must be ON)" OFF)

set(CMAKE_CXX_STANDARD 11)

## COMPILERS
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable") #-Werror
    set(warnings_cuda "-Wall,-Wextra,-Wno-unused-parameter,-Wno-unused-variable") #-Werror
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

if (NOT CONFIGURED_ONCE)
  set(CMAKE_C_FLAGS "${warnings}"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
  # We use -D_GLIBCXX_USE_CXX11_ABI=0 to disable C++11 ABI, since nvcc uses old ABI, generating undefined symbols,
  # while raw g++ generates new ABI without old ABI, causing link time errors when linking host with device code. 
  set(CMAKE_CXX_FLAGS "${warnings} -D_GLIBCXX_USE_CXX11_ABI=0"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()


## DEPENDENCIES
set(Boost_USE_STATIC_LIBS YES)
find_package(Boost 1.63.0 REQUIRED COMPONENTS serialization program_options)
include_directories(${PROJECT_SOURCE_DIR}/src ${Boost_INCLUDE_DIR})
link_directories(${PROJECT_BINARY_DIR}/lib ${Boost_LIBRARY_DIR})

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
option(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -rdc=true --compiler-options "${warnings_cuda}" -std=c++11 -g -G)
# Separate compilation in CUDA is still a bugged feature of CMake.FindCUDA
# https://gitlab.kitware.com/cmake/cmake/issues/15157
set(CUDA_SEPARABLE_COMPILATION ON)

link_directories("/usr/local/cuda/lib64/")

if (COMPR)
    message(STATUS "Compression module enable -- compiling compression interface")
    add_definitions(-DCOMPR)

    if (SIMDCAI)
        # Add external project lemire/SIMDCompressionAndIntersection - this is used for reference implementations of
        # integer compression, delta encoding and intersections of posting lists
        ExternalProject_Add(
          SIMDCAI_PROJ # Name for custom target
          PREFIX "simdcai" # Root dir for entire project
          GIT_REPOSITORY https://github.com/lemire/SIMDCompressionAndIntersection # URL of git repo
          UPDATE_COMMAND "" # Source work-tree update command
          PATCH_COMMAND bash -c "patch --directory=${PROJECT_BINARY_DIR}/simdcai/src/SIMDCAI --strip=1 --forward --quiet < ${PROJECT_SOURCE_DIR}/simdcai.patch $<SEMICOLON> echo OK "  # Patch SIMDCAI_PROJ source for NVCC
          CONFIGURE_COMMAND "" # Build tree configuration command
          BUILD_COMMAND make "-j4" # Command to drive the native build
          BUILD_IN_SOURCE 1 # Use source dir for build dir
          INSTALL_COMMAND "" # Command to drive install after build
          TEST_AFTER_INSTALL 1 # Add test step executed after install step
          TEST_COMMAND "./unit" # Command to drive test
        )

        # Retrieve the in-source build location of SIMDCAI
        ExternalProject_Get_Property(SIMDCAI_PROJ BINARY_DIR)
        set(SIMDCAI_INSTALL_DIR ${BINARY_DIR})
        # include_directories(${PROJECT_SOURCE_DIR}/simdcai/src)
        message(STATUS "SIMDCAI_BINARY_DIR: ${BINARY_DIR}")

        SET(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
        cuda_include_directories(SYSTEM "${SIMDCAI_INSTALL_DIR}/include" ${PROJECT_SOURCE_DIR}/simdcai/src)
    endif()

else()
    message(STATUS "Compression module disabled")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(static)
add_subdirectory(perf)
add_subdirectory(doc)
