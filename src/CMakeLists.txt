macro(install_headers HEADERS)
  foreach ( file ${${HEADERS}} )
    get_filename_component( dir ${file} DIRECTORY )
    install( FILES ${file} DESTINATION include/${dir} )
  endforeach()
endmacro()

configure_file(
  ${PROJECT_SOURCE_DIR}/src/genie/configure.h.in
  ${PROJECT_SOURCE_DIR}/src/genie/configure.h
)

set( GENIE_COMPILATION_UNITS
  genie/execution_policy/single_range.cc
  genie/execution_policy/single_value.cc
  genie/execution_policy/validation.cc
  genie/interface/config.cc
  genie/interface/execution_policy.cc
  genie/interface/io.cc
  genie/interface/search.cc
  genie/matching/heap_count.cu  
  genie/matching/interface.cc
  genie/matching/knn.cu
  genie/matching/match.cu
  genie/matching/match_common.cu
  genie/original/interface.cc
  genie/query/interface.cc
  genie/query/query.cc
  genie/table/interface.cc
  genie/table/inv_list.cc
  genie/table/inv_table.cu
  genie/utility/FileReader.cc
  genie/utility/init.cu
  genie/utility/Logger.cc
  genie/utility/scan.cu
  genie/utility/serialization.cc
  genie/utility/Timing.cc
)

set( GENIE_TOP_LEVEL_HEADER
  genie/genie.h
)

set( GENIE_HEADERS
  genie/exception/exception.h
  genie/execution_policy/single_range.h
  genie/execution_policy/single_value.h
  genie/execution_policy/validation.h
  genie/interface/config.h
  genie/interface/execution_policy.h
  genie/interface/io.h
  genie/interface/search.h
  genie/interface/types.h
  genie/matching/heap_count.h
  genie/matching/interface.h
  genie/matching/knn.h
  genie/matching/match.h
  genie/matching/match_common.h
  genie/original/interface.h
  genie/query/interface.h
  genie/query/query.h
  genie/table/interface.h
  genie/table/inv_list.h
  genie/table/inv_table.h
  genie/utility/cuda_macros.h
  genie/utility/FileReader.h
  genie/utility/init.h
  genie/utility/Logger.h
  genie/utility/PerfLogger.hpp
  genie/utility/scan.h
  genie/utility/serialization.h
  genie/utility/Timing.h
)


if( GENIE_COMPR )
  set( GENIE_COMPRESSION_COMPILATION_UNITS
    genie/table/inv_compr_table.cu
    genie/matching/match_integrated.cu
    genie/compression/DeviceCodecs.cu
    genie/compression/DeviceCodecFactory.cu
    genie/compression/DeviceCompositeCodec.cu
    genie/compression/DeviceSerialCodec.cu
    genie/compression/DeviceBitPackingCodec.cu
    genie/compression/DeviceBitPackingHelpers.cu
    genie/compression/DeviceVarintCodec.cu
  )
  
  set( GENIE_COMPRESSION_HEADERS
    genie/table/inv_compr_table.h
    genie/matching/match_integrated.h
    genie/compression/DeviceCodecs.h
    genie/compression/DeviceCodecFactory.h
    genie/compression/DeviceCompositeCodec.h
    genie/compression/DeviceSerialCodec.h
    genie/compression/DeviceDeltaHelper.h
    genie/compression/DeviceBitPackingCodec.h
    genie/compression/DeviceBitPackingHelpers.h
    genie/compression/DeviceVarintCodec.h
  )
endif()


set(BUILD_SHARED_LIBS OFF)
add_library( ${PROJECT_NAME}
  ${GENIE_COMPILATION_UNITS} ${GENIE_COMPRESSION_COMPILATION_UNITS})

target_link_libraries( ${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${BZIP2_LIBRARIES})

set_target_properties( ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION})


install(TARGETS ${PROJECT_NAME} 
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install_headers(GENIE_TOP_LEVEL_HEADER)
install_headers(GENIE_HEADERS)
install_headers(GENIE_COMPRESSION_HEADERS)
