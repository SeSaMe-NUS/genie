genie_add_simple_executable (example_table_from_file
    example_table_from_file.cc)

genie_add_simple_executable (example_sift
    example_sift.cc)

genie_add_simple_executable (example_tweets
    example_tweets.cc)

genie_add_simple_executable (example_sift_subsequence
    example_sift_subsequence.cc)

genie_add_simple_executable (example_1st_level_interface
	example_1st_level_interface.cc)

genie_add_simple_executable (example_2nd_level_interface
	example_2nd_level_interface.cc)

set(GENIE_STATIC_FILES_FOR_EXAMPLES
    "../static/sift_1k.csv" "../static/sift_1k_query.csv" "../static/tweets_4k.csv")

install(FILES ${GENIE_STATIC_FILES_FOR_EXAMPLES} DESTINATION static)
