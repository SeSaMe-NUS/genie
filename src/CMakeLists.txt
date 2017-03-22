configure_file(
  ${PROJECT_SOURCE_DIR}/src/GPUGenie/configure.h.in
  ${PROJECT_SOURCE_DIR}/src/GPUGenie/configure.h
)

if (USE_MPI)
  find_package(MPI REQUIRED)
  MESSAGE(STATUS "Building GENIE with MPI support")
  option(USE_DYNAMIC "Dynamic update of threshold" OFF)
  include_directories(${MPI_INCLUDE_PATH})
  add_subdirectory(DistGenie)
endif()

set(LIB_SRC_LIST
  GPUGenie.h

  GPUGenie/genie_errors.h
  GPUGenie/genie_errors.cu

  GPUGenie/Logger.h
  GPUGenie/Logger.cc
  
  GPUGenie/Timing.h
  GPUGenie/Timing.cc

  GPUGenie/inv_list.h
  GPUGenie/inv_list.cu
  GPUGenie/inv_table.h
  GPUGenie/inv_table.cu
  GPUGenie/query.h
  GPUGenie/query.cu
  GPUGenie/match.h
  GPUGenie/match.cu
  GPUGenie/interface.h
  GPUGenie/interface.cu
  GPUGenie/knn.h
  GPUGenie/knn.cu
  GPUGenie/heap_count.h
  GPUGenie/heap_count.cu  
  GPUGenie/FileReader.h
  GPUGenie/FileReader.cu
)

cuda_add_library(
  ${PROJECT_NAME}
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

install_header(Logger.h)
install_header(interface.h)
install_header(inv_list.h)
install_header(inv_table.h)
install_header(query.h)
