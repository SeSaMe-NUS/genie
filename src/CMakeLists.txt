include_directories(${PROJECT_SOURCE_DIR}/src)
cuda_include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${PROJECT_SOURCE_DIR}/src)

set(LIB_SRC_LIST
  GaLG.h

  GaLG/lib/libcsv/csv.h
  GaLG/lib/libcsv/libcsv.c

  GaLG/raw_data.h
  GaLG/raw_data.cc
  GaLG/inv_list.h
  GaLG/inv_list.cc

  GaLG/parser/parser.h
  GaLG/parser/csv.cc)

cuda_add_library(
  ${PROJECT_NAME}
  SHARED
  ${LIB_SRC_LIST})

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_API_VERSION}.${PROJECT_MINOR_VERSION}
  SOVERSION ${PROJECT_API_VERSION})


install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(FILES GaLG.h DESTINATION include)

function(install_header DIR TARGET)
  install(FILES GaLG/${DIR}/${TARGET} DESTINATION include/GaLG/${DIR})
endfunction()

install_header(parser parser.h)