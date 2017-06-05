cuda_add_executable(csv2binary csv2binary.cu)
target_link_libraries(csv2binary
	-lpthread
	${PROJECT_NAME}
    -lcudadevrt
	)
