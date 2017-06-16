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

set(GENIE_STATIC_FILES_FOR_EXAMPLES
    "../static/sift_1k.csv" "../static/sift_1k.dat" "../static/sift_1k_query.csv"
    "../static/tweets_4k.csv" "../static/tweets_4k.dat")

install(FILES ${GENIE_STATIC_FILES_FOR_EXAMPLES} DESTINATION static)
