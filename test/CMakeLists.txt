function(genie_add_test_case TARGET)
   add_executable(${TARGET} ${ARGN})
   add_dependencies(${TARGET} ${PROJECT_NAME})
   set_property(TARGET ${TARGET} PROPERTY LINK_DEPENDS ${PROJECT_BINARY_DIR}/lib/libGPUGenie.a)  
   target_link_libraries(${TARGET}
                         ${PROJECT_NAME}
                         ${MPI_LIBRARIES}
                         ${CUDA_cudadevrt_LIBRARY})
   add_test(${TARGET} ${PROJECT_BINARY_DIR}/bin/${TARGET})
endfunction()

genie_add_test_case(test_csv_input_sift_data
    test_csv_input_sift_data.cu)

genie_add_test_case(test_binary_input_sift_data
    test_binary_input_sift_data.cu)

genie_add_test_case(test_csv_input_multiload
    test_csv_input_multiload.cu)

genie_add_test_case(test_binary_input_multiload
    test_binary_input_multiload.cu)

genie_add_test_case(test_save2gpu_sift
    test_save2gpu_sift.cu)

genie_add_test_case(test_serialization_for_table
    test_serialization_for_table.cu)

genie_add_test_case(test_csv_input_tweets
    test_csv_input_tweets.cu)

genie_add_test_case(test_binary_input_tweets
    test_binary_input_tweets.cu)

genie_add_test_case(test_csv_multiload_tweets
    test_csv_multiload_tweets.cu)

genie_add_test_case(test_binary_multiload_tweets
    test_binary_multiload_tweets.cu)

genie_add_test_case(test_subsequence_search_csv_input
    test_subsequence_search_csv_input.cu)

genie_add_test_case(test_subsequence_search_binary_input
    test_subsequence_search_binary_input.cu)

genie_add_test_case(test_hashtable_overflow
    test_hashtable_overflow.cu)

genie_add_test_case(test_multiple_search
    test_multiple_search.cu)

<<<<<<< master
genie_add_test_case(test_refactoring_new_interface
	test_refactoring_new_interface.cu)
=======
genie_add_test_case(test_serialization
    test_serialization.cu)


>>>>>>> Testing inplementation of free boost serializer on inv_table

if (GENIE_COMPR)
    genie_add_test_case(test_compression_naive
        test_compression_naive.cu)

    genie_add_test_case(test_compression_codecs
        test_compression_codecs.cu)

    genie_add_test_case(test_compression_kernels
        test_compression_kernels.cu)

    if (GENIE_SIMDCAI)
        # Bug in target_compile_options for NVCC targets
        # (not sure if newer version of CMake resolved this)

        # This one seems to be ignored by NVCC -- NVCC does not use COMPILER_FLAGS
        # target_compile_options(test_compression_test PUBLIC "--compiler-options -mssse3")

        # This one does not work as well -- probably CMake.CUDA bug
        # https://cmake.org/pipermail/cmake/2012-September/052007.html
        # set_target_properties(test_compression_test PROPERTIES
        #     CUDA_NVCC_FLAGS "--compiler-options -mssse3")
        
        # Relatively complicated workaround:
        #       http://stackoverflow.com/questions/24560001/how-to-set-cmake-cuda-nvcc-flags-per-cu-file
        set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options -mssse3)

        genie_add_test_case(test_compression_test
            test_compression_test.cu)

        genie_add_test_case(test_compression_query
            test_compression_query.cu)

        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})

        target_link_libraries(test_compression_test
            ${SIMDCAI_PROJ_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
        target_link_libraries(test_compression_query
            ${SIMDCAI_PROJ_INSTALL_DIR}/libSIMDCompressionAndIntersection.a)
    endif()

endif()
