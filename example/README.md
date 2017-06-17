This folder contains examples for GPU search

`example.cu`: This is an exapmle for the kNN search on the SIFT feature data, one line for one SIFT features

`example_tweets.cu`: This is an exaple for kNN search on the tweets data, one line is for one tweet. The interger of each line represents a encoded word

`example_adaptiveThreshold.cu`: This is an exaple for kNN search with adaptiveThreshold. A Count Heap method is use to support large number of queries for top-k query

`*_binary.cu`: All files whose name end like this are merely different from above files on the input. The data input is from binary files.
