include_directories(${MPI_INCLUDE_PATH})
cuda_add_executable(dgenie main.cu)
target_link_libraries(dgenie
	${MPI_LIBRARIES}
	-l${PROJECT_NAME}
	${Boost_LIBRARIES}
	)
