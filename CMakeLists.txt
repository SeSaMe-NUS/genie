## GENERAL PROJECT SETTINGS

cmake_minimum_required(VERSION 3.7)

set(PROJECT_NAME GPUGenie)
set(PROJECT_API_VERSION 1)
set(PROJECT_MINOR_VERSION 0.0)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
project(${PROJECT_NAME})
enable_testing()
include(CheckCXXCompilerFlag)
include(ExternalProject)

option(GENIE_COMPR "Build compression module" ON)
option(GENIE_SIMDCAI "Build SIMDCAIR library (from Daniel Lemire) and related tests (COMPR must be ON)" OFF)
option(GENIE_DISTRIBUTED "Build distributed GENIE" OFF)
option(GENIE_EXAMPLES "Build GENIE examples" ON)
option(BUILD_SHARED_LIBS OFF)

# If the user does not specify PREFIX, we default to ${CMAKE_BINARY_DIR} instead of system directory /usr
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Default install path (prefix)" FORCE)
endif()


## COMPILERS

set(CMAKE_CXX_STANDARD 11)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable") #-Werror
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

set(CMAKE_C_FLAGS "${warnings}" CACHE STRING "Flags used by the compiler during all build types." FORCE)
set(CMAKE_CXX_FLAGS "${warnings}" CACHE STRING "Flags used by the compiler during all build types." FORCE)


## DEPENDENCIES

### Boost

set(Boost_USE_STATIC_LIBS ON)
if (GENIE_COMPR)
  set(Boost_COMPONETNS serialization program_options)
else()
  set(Boost_COMPONETNS serialization)
endif()
# WARNING: CMake may issue a warning: Imported targets not available for Boost version 106400
# This happens when CMake doesn't recognize too new version of Boost, for example CMake 3.7 doesn't recognize Boost
# 1.64, but CMake 3.8 already does. The warning causes missing Boost components dependencies, but may be ignored.
find_package(Boost 1.63.0 REQUIRED COMPONENTS ${Boost_COMPONETNS})

### MPI & OpenMP

if (GENIE_DISTRIBUTED)
  find_package(MPI REQUIRED)
  find_package(OpenMP REQUIRED)
  MESSAGE(STATUS "Building GENIE with MPI support")
  include_directories(${MPI_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

### CUDA

find_package(CUDA REQUIRED)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS --std c++11 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_52,code=sm_52 --relocatable-device-code true --Werror cross-execution-space-call)

# BUG: with CUDA_NVCC_FLAGS_CONFIG variable processing in CMake
# https://gitlab.kitware.com/cmake/cmake/issues/16411
# Workaround is to specify CUDA_NVCC_FLAGS manually based on build type
# (Issue still present in CMake 3.8.2)
# set(CUDA_NVCC_FLAGS_DEBUG  --debug --device-debug --generate-line-info)
# set(CUDA_NVCC_FLAGS_RELEASE --optimize 3 --generate-line-info -DNDEBUG) 
# set(CUDA_NVCC_FLAGS_RELWITHDEBUGINFO --generate-line-info)
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --debug --device-debug --generate-line-info)
elseif(CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --optimize 3 --generate-line-info -DNDEBUG) 
else()
  message(FATAL "Unknown build type, cannot determine NVCC flags")
endif()

set(CUDA_SEPARABLE_COMPILATION ON)


## COMPRESSION MODULE

if (GENIE_COMPR)
    message(STATUS "Compression module enabled")

    if (GENIE_SIMDCAI)
        # Add external project lemire/SIMDCompressionAndIntersection - this is used for reference implementations of
        # integer compression, delta encoding and intersections of posting lists
        ExternalProject_Add(
          SIMDCAI_PROJ # Name for custom target
          PREFIX "simdcai" # Root dir for entire project
          GIT_REPOSITORY https://github.com/lemire/SIMDCompressionAndIntersection # URL of git repo
          UPDATE_COMMAND "" # Source work-tree update command
          PATCH_COMMAND bash -c "patch --directory=${PROJECT_BINARY_DIR}/simdcai/src/SIMDCAI_PROJ --strip=1 --forward --quiet < ${PROJECT_SOURCE_DIR}/simdcai.patch"  # Patch SIMDCAI_PROJ source for NVCC
          CONFIGURE_COMMAND "" # Build tree configuration command
          BUILD_COMMAND make "-j4" # Command to drive the native build
          BUILD_IN_SOURCE 1 # Use source dir for build dir
          INSTALL_COMMAND "" # Command to drive install after build
          TEST_AFTER_INSTALL 1 # Add test step executed after install step
          TEST_COMMAND "./unit" # Command to drive test
        )

        # Retrieve the in-source build location of SIMDCAI
        ExternalProject_Get_Property(SIMDCAI_PROJ BINARY_DIR)
        set(SIMDCAI_PROJ_INSTALL_DIR ${BINARY_DIR})

        SET(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
        cuda_include_directories(SYSTEM "${SIMDCAI_PROJ_INSTALL_DIR}/include" ${PROJECT_SOURCE_DIR}/simdcai/src)
    endif()
else()
    message(STATUS "Compression module disabled")
    if (GENIE_SIMDCAI)
        message(SEND_ERROR "SIMDCAI cannot be enabled without COMPR!" )
    endif()
endif()


## INCLUDES AND LINK DIRECTORIES

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${Boost_INCLUDE_DIR})
link_directories(${PROJECT_BINARY_DIR}/lib ${Boost_LIBRARY_DIR})


## HELPER FUNCTIONS

function(genie_add_simple_executable TARGET)
    cuda_add_executable(${TARGET} ${ARGN})
    add_dependencies(${TARGET} ${PROJECT_NAME})
    set_property(TARGET ${TARGET} PROPERTY LINK_DEPENDS ${PROJECT_BINARY_DIR}/lib/libGPUGenie.a)  
    target_link_libraries(${TARGET}
        ${PROJECT_NAME}
        ${CUDA_cudadevrt_LIBRARY})
    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin)
endfunction()


## OUTPUT OPTIONS

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


## CMAKE SUBDIRECTORIES

add_subdirectory(doc)
if (GENIE_EXAMPLES)
  add_subdirectory(example)
endif()
add_subdirectory(perf)
add_subdirectory(src)
if (GENIE_DISTRIBUTED)
  add_subdirectory(src/DistGenie)
endif()
add_subdirectory(static)
add_subdirectory(test)
add_subdirectory(utility)
