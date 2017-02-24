find_package(Doxygen)

if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/doc/doxy.in
		${PROJECT_BINARY_DIR}/Doxyfile
		@ONLY)
	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
		COMMENT "Generating documentation with Doxygen"
		VERBATIM)
endif(DOXYGEN_FOUND)
