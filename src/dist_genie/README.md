# Overview of DistGenie

DistGenie is a program built on top of the GENIE library to support multi-GPU and multi-node
similarity search. It has the following components

- online.cu (the program entry)
- file.cu (I/O related operations)
- parser.cu (Query & config file parser)
- scheduler.cc (Query listenser & scheduler)
- search.cu (calls GENIE to perform similarity search)
- sorting.cc (handles sorting of final output)
