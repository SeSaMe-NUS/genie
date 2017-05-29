cuda_include_directories(${PROJECT_SOURCE_DIR}/src ${Boost_INCLUDE_DIRS})

set(LIB_SRC_LIST
  GPUGenie.h
  

  GPUGenie/genie_errors.h
  GPUGenie/genie_errors.cu
  GPUGenie/inv_list.h
  GPUGenie/inv_list.cu
  GPUGenie/inv_table.h
  GPUGenie/inv_table.cu
  GPUGenie/inv_compr_table.h
  GPUGenie/inv_compr_table.cu
  GPUGenie/query.h
  GPUGenie/query.cu
  GPUGenie/match.h
  GPUGenie/match.cu
  GPUGenie/match_sepkernel.h
  GPUGenie/match_sepkernel.cu
  GPUGenie/match_integrated.h
  GPUGenie/match_integrated.cu
  GPUGenie/interface.h
  GPUGenie/interface.cu
  GPUGenie/knn.h
  GPUGenie/knn.cu
  GPUGenie/heap_count.h
  GPUGenie/heap_count.cu  
  GPUGenie/FileReader.h
  GPUGenie/FileReader.cu
  GPUGenie/DeviceCodecs.h
  GPUGenie/DeviceCodecs.cu
  GPUGenie/DeviceCompositeCodec.h
  GPUGenie/DeviceCompositeCodec.cu
  GPUGenie/DeviceSerialCodec.h
  GPUGenie/DeviceSerialCodec.cu
  GPUGenie/DeviceDeltaHelper.h
  GPUGenie/DeviceBitPackingCodec.h
  GPUGenie/DeviceBitPackingCodec.cu
  GPUGenie/DeviceBitPackingHelpers.h
  GPUGenie/DeviceBitPackingHelpers.cu
  GPUGenie/DeviceVarintCodec.h
  GPUGenie/DeviceVarintCodec.cu
  GPUGenie/scan.h
  GPUGenie/scan.cu
  GPUGenie/PerfLogger.hpp
  GPUGenie/Logger.h
  GPUGenie/Logger.cpp
  GPUGenie/Timing.h
  GPUGenie/Timing.cpp
)

cuda_add_library(
  ${PROJECT_NAME}
  STATIC
  ${LIB_SRC_LIST})

target_link_libraries(
  ${PROJECT_NAME} -lboost_serialization)

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

install_header(Logger.h)
install_header(interface.h)
install_header(inv_list.h)
install_header(inv_table.h)
install_header(query.h)
