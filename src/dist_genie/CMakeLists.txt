include_directories(${PROJECT_SOURCE_DIR}/src/dist_genie)

genie_add_simple_executable(odgenie
    online.cu search.cu parser.cu file.cu scheduler.cc sorting.cc)

target_link_libraries(odgenie
    ${MPI_LIBRARIES})
