include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/simdcai/src)
if (COMPR)
    SET(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
    include_directories(SYSTEM "${SIMDCAI_INSTALL_DIR}/include")
    include_directories("/usr/local/cuda/samples/common/inc /usr/local/lib")
endif()

link_directories(${Boost_LIBRARY_DIR})
link_directories(${PROJECT_SOURCE_DIR}/lib /usr/local/lib)


function(add_test_case TARGET)
   cuda_add_executable(${TARGET} ${ARGN})
   add_dependencies(${TARGET} ${PROJECT_NAME})
   target_link_libraries(${TARGET}
                         -lpthread
                         -lboost_serialization
                         ${PROJECT_SOURCE_DIR}/lib/libGPUGenie.a
                         -lcudadevrt)
   add_test(${TARGET} ${PROJECT_BINARY_DIR}/bin/${TARGET})
endfunction()

add_test_case(test_1_csv_input_sift_data test_1.cu)
add_test_case(test_2_binary_input_sift_data test_2.cu)
add_test_case(test_3_csv_input_multiload test_3.cu)
add_test_case(test_4_binary_input_multiload test_4.cu)
add_test_case(test_5_save2gpu_sift test_5.cu)
add_test_case(test_6_serialization_for_table test_6.cu)
add_test_case(test_7_csv_input_tweets test_7.cu)
add_test_case(test_8_binary_input_tweets test_8.cu)
add_test_case(test_9_csv_multiload_tweets test_9.cu)
add_test_case(test_10_binary_multiload_tweets test_10.cu)
add_test_case(test_11_subsequence_search_csv_input test_11.cu)
add_test_case(test_12_subsequence_search_binary_input test_12.cu)
add_test_case(test_16_hashtable_overflow test_16.cu)


if (COMPR)
    # # This is possibly a bug in CMake - target_link_libraries does not add -I<path> parameter to nvcc
    # # Workaround: add global include_directories instead
    #
    # target_include_directories(test_13_compression_test PUBLIC
    #     # ${PROJECT_SOURCE_DIR}/fastpfor/headers
    #     ${SIMDCAI_INSTALL_DIR}/include)
    # target_link_libraries(test_13_compression_test
    #     ${PROJECT_SOURCE_DIR}/fastpfor/libFastPFor.a

    # This one seems to be ignored by NVCC -- NVCC does not use COMPILER_FLAGS
    # target_compile_options(test_13_compression_test PUBLIC "--compiler-options -mssse3")

    # This one does not work as well -- probably CMake.CUDA bug
    # https://cmake.org/pipermail/cmake/2012-September/052007.html
    # set_target_properties(test_13_compression_test PROPERTIES
    #     CUDA_NVCC_FLAGS "--compiler-options -mssse3")
    # Relatively complicated workaround:
    #       http://stackoverflow.com/questions/24560001/how-to-set-cmake-cuda-nvcc-flags-per-cu-file

    set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options -mssse3)

    add_test_case(test_13_compression_test test_13.cu)
    add_test_case(test_14_compression_query test_14.cu)
    add_test_case(test_15_compression_naive test_15.cu)
    add_test_case(test_17_scan4 test_17.cu)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})

    target_link_libraries(test_13_compression_test
        ${SIMDCAI_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
    target_link_libraries(test_14_compression_query
        ${SIMDCAI_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
    target_link_libraries(test_15_compression_naive
        ${SIMDCAI_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
    target_link_libraries(test_17_scan4
        ${SIMDCAI_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
endif()
