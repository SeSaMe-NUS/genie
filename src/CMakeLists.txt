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
  GaLG/inv_table.h
  GaLG/inv_table.cc
  GaLG/query.h
  GaLG/query.cc

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

function(install_header TARGET)
  install(FILES GaLG/${TARGET} DESTINATION include/GaLG)
endfunction()

install(FILES GaLG/parser/parser.h DESTINATION include/GaLG/parser)
install_header(raw_data.h)
install_header(inv_list.h)
install_header(inv_table.h)