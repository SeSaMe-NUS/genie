## GENERAL PROJECT SETTINGS

cmake_minimum_required(VERSION 3.8)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
project("GPUGenie"
  VERSION 0.1
  LANGUAGES C CXX CUDA)
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
if (NOT ${CMAKE_COMPILER_IS_GNUCXX})
  message(FATAL_ERROR "Only GNU compilers are supported")
endif()

set(genie_warnings "-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable") #-Werror
set(CMAKE_C_FLAGS "${genie_warnings}")
set(CMAKE_CXX_FLAGS "${genie_warnings}")

set(CMAKE_CUDA_FLAGS "--std c++11 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_52,code=sm_52 --relocatable-device-code true --Werror cross-execution-space-call")
# Disable cudafe warning: "type qualifier is meaningless on cast type" occuring in boost::archive
# https://stackoverflow.com/questions/14831051/how-to-disable-compiler-warnings-with-nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=cast_to_qualified_type\"")
# Boost 1.64 has a bug with Cuda 7.5 and 8.0, where compilation fails on static assert from g++ standard library 
# https://svn.boost.org/trac10/ticket/13049
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX11_NOEXCEPT")
set(CMAKE_CUDA_FLAGS_DEBUG  "--debug --device-debug --generate-line-info")
set(CMAKE_CUDA_FLAGS_RELEASE "--optimize 3 --generate-line-info -DNDEBUG")
# set(CUDA_NVCC_FLAGS_RELWITHDEBUGINFO "--optimize 2 --generate-line-info")

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_SEPARABLE_COMPILATION ON)


## DEPENDENCIES

### Boost

set(Boost_USE_STATIC_LIBS ON)
set(Boost_COMPONETNS iostreams serialization program_options)
# WARNING: CMake may issue a warning: Imported targets not available for Boost version 106400
# This happens when CMake doesn't recognize too new version of Boost, for example CMake 3.7 doesn't recognize Boost
# 1.64, but CMake 3.8 already does. The warning causes missing Boost components dependencies, but may be ignored.
find_package(Boost 1.63.0 REQUIRED COMPONENTS ${Boost_COMPONETNS})

### MPI & OpenMP

if (GENIE_DISTRIBUTED)
  # WORKAROUND: Unset all variables for FindMPI. There are no input variables for FindMPI for MPI home directory.
  # Unsetting MPI_HOME (undocumented variable FindMPI uses) is not sufficient.
  set(_FINDMPI_CACHED_VARIABLES MPIEXEC MPIEXEC_MAX_NUMPROCS MPIEXEC_NUMPROC_FLAG MPIEXEC_POSTFLAGS MPIEXEC_PREFLAGS
    MPI_CXX_COMPILER MPI_CXX_COMPILE_FLAGS MPI_CXX_INCLUDE_PATH MPI_CXX_LIBRARIES MPI_CXX_LINK_FLAGS MPI_C_COMPILER 
    MPI_C_COMPILE_FLAGS MPI_C_INCLUDE_PATH MPI_C_LIBRARIES MPI_C_LINK_FLAGS MPI_EXTRA_LIBRARY MPI_LIBRARY)
  foreach (_FINDMPI_VAR ${_FINDMPI_CACHED_VARIABLES})
    unset(${_FINDMPI_VAR} CACHE)
  endforeach()
  
  find_package(MPI REQUIRED) # Use MPI_HOME or env(MPI_HOME) to define root directory of MPI installation
  unset(MPI_HOME CACHE)
  message(STATUS "Found mpiexec: ${MPIEXEC}")

  # Check for correct OpenMPI and version 
  if (NOT MPIEXEC OR NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "OpenMPI not found!")
  endif()
  execute_process(COMMAND ${MPI_C_COMPILER} --showme:version OUTPUT_VARIABLE MPI_VERSION)
  if (NOT ${MPI_VERSION} MATCHES ".*Open MPI.*" OR
      NOT ${MPI_VERSION} MATCHES ".* 1\\.[7-9]\\..*| .*1\\.10\\..*| .* 2\\.[0-9]+\\..*")
    message(FATAL_ERROR "MPI not OpenMPI, or OpenMPI has unknown or too low version! Required OpenMPI 1.7 or higher.")
  endif()

  find_package(OpenMP REQUIRED)

  message(STATUS "Distributed GENIE module enabled")
  include_directories(${MPI_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## Bzip2

find_package(BZip2 REQUIRED)

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
        include_directories(SYSTEM "${SIMDCAI_PROJ_INSTALL_DIR}/include" ${PROJECT_SOURCE_DIR}/simdcai/src)
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
    add_executable(${TARGET} ${ARGN})
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
add_subdirectory(src)
if (GENIE_DISTRIBUTED)
  add_subdirectory(src/DistGenie)
endif()
add_subdirectory(src/perftoolkit)
add_subdirectory(static)
add_subdirectory(test)
add_subdirectory(utility)
