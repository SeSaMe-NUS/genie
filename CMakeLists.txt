cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME GaLG)
set(PROJECT_API_VERSION 1)
set(PROJECT_MINOR_VERSION 0.0)

project(${PROJECT_NAME})
enable_testing()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()
if (NOT CONFIGURED_ONCE)
  set(CMAKE_C_FLAGS "${warnings}"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
  set(CMAKE_CXX_FLAGS "${warnings}"
    CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()


find_package(CUDA QUIET REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)
