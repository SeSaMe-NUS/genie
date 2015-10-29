libGaLG
===
libGaLG is a fast GPU inverted list searching library. It builds the database from a csv file or a vector of instances. Then libGaLG will consturct the inverted list table and transfer it to the device. libGaLG provides a simple way to perform the k match queries. User may define queries and their matching ranges, then directly call the matching funtion. The library will parallel process all queries and save the matching result into a device_vector. A top k search can also be simply perfromed. libGaLG uses parallel searching to determine the top k values in a vector. It is much faster than the CPU searching algorithm. All device methods are wrapped in host methods. Developers are not required to configure the device function call.


### Compiler and development

You are required to install gcc, g++, nvcc and cmake. Please make sure that the cmake version is greater tha nversion 2.8 and cuda 7.

To compile the program, just type 

```
make
```

The library will be generated within the folder "build"

After compiling, you can go to folder "example", and type "make". Some running examples will be compiled fnd generated for testing.

### Install

You are required to install gcc, g++, nvcc and cmake. Please make sure that the cmake version is greater than version 2.8 and cuda 7.

To install the libGaLG, directlly call the install script, `install.sh`.


### Running example
You can see a running example in the folder "example".
Just command:

```cpp
cd example
make
```

and then command

```
 ./example_bin
```


You can see the query results based on the data file "sift_1k.csv". 
The exaple also gives an comprehensive description about the parameter. 

How to use the library is also shown in the the file /example/makefile

### log
2015.09.10, add running example

2015.09.15, merge with branch "fixtopk" 

2015.09.15, change branch "runningDemo" as "master"

2015.10.29, add adaptive threshold method for topk search

