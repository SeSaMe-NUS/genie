include_directories(${PROJECT_SOURCE_DIR}/src)
cuda_include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${PROJECT_SOURCE_DIR}/src)

set(LIB_SRC_LIST
  GaLG.h

  lib/libcsv/csv.h
  lib/libcsv/libcsv.c

  raw_data.h
  raw_data.cc

  tool/converter.h
  tool/parser.h
  tool/s2.cc
  tool/csv.cc)

cuda_add_library(
  ${PROJECT_NAME}
  SHARED
  ${LIB_SRC_LIST})

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
  VERSION ${PROJECT_API_VERSION}.${PROJECT_MINOR_VERSION}
  SOVERSION ${PROJECT_API_VERSION})
