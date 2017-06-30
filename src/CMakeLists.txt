configure_file(
	${PROJECT_SOURCE_DIR}/src/GPUGenie/configure.h.in
	${PROJECT_SOURCE_DIR}/src/GPUGenie/configure.h
)

set( GENIE_COMPILATION_UNITS
  GPUGenie/genie_errors.cu
  GPUGenie/inv_list.cu
  GPUGenie/inv_table.cu
  GPUGenie/query.cu
  GPUGenie/match.cu
  GPUGenie/match_common.cu
  GPUGenie/interface.cu
  GPUGenie/knn.cu
  GPUGenie/heap_count.cu  
  GPUGenie/FileReader.cu
  GPUGenie/scan.cu
  GPUGenie/Logger.cu
  GPUGenie/Timing.cpp
  GPUGenie/interface/config.cc
  GPUGenie/interface/execution_policy.cu
  GPUGenie/interface/search.cu
  GPUGenie/interface/io.cu
  GPUGenie/execution_policy/single_value.cu
  GPUGenie/execution_policy/single_range.cu
  GPUGenie/exception/exception.cc
  GPUGenie/table/interface.cu
  GPUGenie/query/interface.cu
  GPUGenie/matching/interface.cu
)

set( GENIE_TOP_LEVEL_HEADER
  GPUGenie.h
)

set( GENIE_HEADERS
  GPUGenie/genie_errors.h
  GPUGenie/inv_list.h
  GPUGenie/inv_table.h
  GPUGenie/query.h
  GPUGenie/match.h
  GPUGenie/match_common.h
  GPUGenie/interface.h
  GPUGenie/knn.h
  GPUGenie/heap_count.h
  GPUGenie/FileReader.h
  GPUGenie/scan.h
  GPUGenie/PerfLogger.hpp
  GPUGenie/Logger.h
  GPUGenie/Timing.h
)


if( GENIE_COMPR )
  set( GENIE_COMPRESSION_COMPILATION_UNITS
    GPUGenie/inv_compr_table.cu
    GPUGenie/match_integrated.cu
    GPUGenie/DeviceCodecs.cu
    GPUGenie/DeviceCodecFactory.cu
    GPUGenie/DeviceCompositeCodec.cu
    GPUGenie/DeviceSerialCodec.cu
    GPUGenie/DeviceBitPackingCodec.cu
    GPUGenie/DeviceBitPackingHelpers.cu
    GPUGenie/DeviceVarintCodec.cu
  )
  
  set( GENIE_COMPRESSION_HEADERS
    GPUGenie/inv_compr_table.h
    GPUGenie/match_integrated.h
    GPUGenie/DeviceCodecs.h
    GPUGenie/DeviceCodecFactory.h
    GPUGenie/DeviceCompositeCodec.h
    GPUGenie/DeviceSerialCodec.h
    GPUGenie/DeviceDeltaHelper.h
    GPUGenie/DeviceBitPackingCodec.h
    GPUGenie/DeviceBitPackingHelpers.h
    GPUGenie/DeviceVarintCodec.h
  )
endif()


set(BUILD_SHARED_LIBS OFF)
add_library( ${PROJECT_NAME}
  ${GENIE_COMPILATION_UNITS} ${GENIE_COMPRESSION_COMPILATION_UNITS})

target_link_libraries( ${PROJECT_NAME}
  ${Boost_LIBRARIES})

set_target_properties( ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION})


install(TARGETS ${PROJECT_NAME} 
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)

install(FILES ${GENIE_TOP_LEVEL_HEADER} DESTINATION include)
install(FILES ${GENIE_HEADERS} DESTINATION include/GPUGenie)
install(FILES ${GENIE_COMPRESSION_HEADERS} DESTINATION include/GPUGenie)
