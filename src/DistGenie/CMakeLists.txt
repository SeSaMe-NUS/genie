include_directories(${PROJECT_SOURCE_DIR}/src/DistGenie)

cuda_add_executable(dgenie offline.cu)
target_link_libraries(dgenie
	${MPI_LIBRARIES}
	${PROJECT_NAME}
	${Boost_LIBRARIES}
	)

cuda_add_executable(odgenie online.cu search.cu parser.cu file.cu)
target_link_libraries(odgenie
	${MPI_LIBRARIES}
	${PROJECT_NAME}
	${Boost_LIBRARIES}
	)
