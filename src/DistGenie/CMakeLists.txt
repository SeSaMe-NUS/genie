include_directories(${MPI_INCLUDE_PATH})
include_directories(${PROJECT_SOURCE_DIR}/src/DistGenie)

cuda_add_executable(dgenie offline.cu)
target_link_libraries(dgenie
	${MPI_LIBRARIES}
	-l${PROJECT_NAME}
	${Boost_LIBRARIES}
	)

cuda_add_executable(odgenie online.cu search.cu parser.cu)
target_link_libraries(odgenie
	${MPI_LIBRARIES}
	-l${PROJECT_NAME}
	${Boost_LIBRARIES}
	)
