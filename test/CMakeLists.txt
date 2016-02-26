include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${PROJECT_SOURCE_DIR}/src)

function(add_test_case TARGET)
  cuda_add_executable(${TARGET} ${ARGN})
  target_link_libraries(${TARGET} 
			${PROJECT_NAME} 
			pthread)

  add_test(${TARGET} ${PROJECT_BINARY_DIR}/bin/${TARGET})
endfunction()

#add_test_case(test_1 test_1.cu)
#add_test_case(test_raw_data test_raw_data.cc)
#add_test_case(test_inv_list test_inv_list.cc)
#add_test_case(test_inv_table test_inv_table.cc)
#add_test_case(test_match test_match.cu)
