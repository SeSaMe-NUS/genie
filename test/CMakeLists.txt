if (USE_MPI)
function(add_test_case TARGET)
   cuda_add_executable(${TARGET} ${ARGN})
   add_dependencies(${TARGET} ${PROJECT_NAME})
   target_link_libraries(${TARGET}
                         -lpthread
						 ${MPI_LIBRARIES}
                         ${PROJECT_NAME}
						 ${Boost_LIBRARIES})
   add_test(${TARGET} ${PROJECT_BINARY_DIR}/bin/${TARGET})
endfunction()
else()
function(add_test_case TARGET)
   cuda_add_executable(${TARGET} ${ARGN})
   add_dependencies(${TARGET} ${PROJECT_NAME})
   target_link_libraries(${TARGET}
                         -lpthread
                         ${PROJECT_NAME}
						 ${Boost_LIBRARIES})
   add_test(${TARGET} ${PROJECT_BINARY_DIR}/bin/${TARGET})
endfunction()
endif()

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
add_test_case(test_17_multiple_search test_17.cu)
