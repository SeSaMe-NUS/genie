include_directories(${PROJECT_SOURCE_DIR}/src)
cuda_include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${PROJECT_SOURCE_DIR}/src)

set(LIB_SRC_LIST
  GPUGenie.h
  
  GPUGenie/lib/bucket_topk/assign_bucket.inc
  GPUGenie/lib/bucket_topk/bucket_topk.h
  GPUGenie/lib/bucket_topk/bucket_topk.inc
  GPUGenie/lib/bucket_topk/count_sum.inc
  GPUGenie/lib/bucket_topk/result_thread_partition.inc
  GPUGenie/lib/bucket_topk/save_result.inc
  GPUGenie/lib/bucket_topk/update_min_max.inc
  GPUGenie/lib/bucket_topk/utils.inc

  GPUGenie/genie_errors.h
  GPUGenie/genie_errors.cu

  GPUGenie/Logger.h
  GPUGenie/Logger.cc
  
  GPUGenie/Timing.h
  GPUGenie/Timing.cc

  GPUGenie/raw_data.h
  GPUGenie/raw_data.cc
  GPUGenie/inv_list.h
  GPUGenie/inv_list.cc
  GPUGenie/inv_table.h
  GPUGenie/inv_table.cu
  GPUGenie/query.h
  GPUGenie/query.cc
  GPUGenie/match.h
  GPUGenie/match.cu
  GPUGenie/topk.h
  GPUGenie/topk.cu
  GPUGenie/interface.h
  GPUGenie/interface.cu
  GPUGenie/knn.h
  GPUGenie/knn.cu
  GPUGenie/FileReader.h
  GPUGenie/FileReader.cc
)

cuda_add_library(
  ${PROJECT_NAME}
  STATIC
  ${LIB_SRC_LIST})


set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_API_VERSION}.${PROJECT_MINOR_VERSION}
  SOVERSION ${PROJECT_API_VERSION})


install(TARGETS ${PROJECT_NAME} 
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
install(FILES GPUGenie.h DESTINATION include)

function(install_header TARGET)
  install(FILES GPUGenie/${TARGET} DESTINATION include/GPUGenie)
endfunction()

install(FILES GPUGenie/parser/parser.h DESTINATION include/GPUGenie/parser)
install_header(Logger.h)
install_header(interface.h)
install_header(raw_data.h)
install_header(inv_list.h)
install_header(inv_table.h)
install_header(query.h)
