
function(genie_add_simple_executable TARGET)
   cuda_add_executable(${TARGET} ${ARGN})
   add_dependencies(${TARGET} ${PROJECT_NAME})
   set_property(TARGET ${TARGET} PROPERTY LINK_DEPENDS ${PROJECT_BINARY_DIR}/lib/libGPUGenie.a)  
   target_link_libraries(${TARGET}
        ${PROJECT_NAME}
        -lcudadevrt)
endfunction()

# Deprecated
genie_add_simple_executable (example_sift_binary
    example_sift_binary.cu)

genie_add_simple_executable (example_table_from_file
    example_table_from_file.cu)

genie_add_simple_executable (example_sift
    example_sift.cu)

# Deprecated
genie_add_simple_executable (example_tweets_binary
    example_tweets_binary.cu)

# Outdated interface
# genie_add_simple_executable (example_sift_multi_table
#     example_sift_multi_table.cu)

genie_add_simple_executable (example_tweets
    example_tweets.cu)

genie_add_simple_executable (example_sift_subsequence
    example_sift_subsequence.cu)

# Deprecated
genie_add_simple_executable (convert_csv2dat
    convert_csv2binary.cu)


