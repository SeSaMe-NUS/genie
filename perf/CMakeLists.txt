
if (COMPR)
    cuda_add_executable(compression compression.cu)
    add_dependencies(compression ${PROJECT_NAME})
    set_property(TARGET compression PROPERTY LINK_DEPENDS ${PROJECT_BINARY_DIR}/lib/libGPUGenie.a)  
    target_link_libraries(compression
        -lpthread -L${BOOST_LIBRARY_DIR} -lboost_serialization -lboost_program_options -l${PROJECT_NAME} -lcudadevrt)
endif()
