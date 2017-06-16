find_package(Doxygen)

if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/doc/doxy.in
		${PROJECT_BINARY_DIR}/doc/Doxyfile
		@ONLY)

	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
		WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc
		COMMENT "Generating documentation with Doxygen"
		VERBATIM)

    # Trailing / in ${CMAKE_CURRENT_BINARY_DIR}/html/ gets rid of the extra html dir in install/doc
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION doc)
endif()
