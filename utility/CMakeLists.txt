cuda_add_executable(csv2binary
    csv2binary.cu)
target_link_libraries(csv2binary
    ${PROJECT_NAME}
    -lcudadevrt)

# Deprecated
genie_add_simple_executable (csv2dat
    csv2dat.cu)
target_link_libraries(csv2dat
	${PROJECT_NAME}
    -lcudadevrt)
