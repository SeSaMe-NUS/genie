include_directories(${MPI_INCLUDE_PATH})

cuda_add_executable(dgenie offline.cu)
target_link_libraries(dgenie
	${MPI_LIBRARIES}
	-l${PROJECT_NAME}
	${Boost_LIBRARIES}
	)

cuda_add_executable(odgenie online.cu)
target_link_libraries(odgenie
	${MPI_LIBRARIES}
	-l${PROJECT_NAME}
	${Boost_LIBRARIES}
	)
