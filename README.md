GPUGenie
===
GPUGenie is a Generic Inverted Index on the GPU. It builds the database from a csv file or a vector of instances. Then GPUGenie will consturct the inverted list table and transfer it to the device. GPUGenie provides a simple way to perform the similarity queries. User may define queries and their matching ranges, then directly call the matching funtion. The library will parallel process all queries and save the matching result into a device_vector. A top k search can also be simply perfromed. GPUGenie uses parallel searching to determine the top k values in a vector. It is much faster than the CPU searching algorithm. All device methods are wrapped in host methods. Developers are not required to configure the device function call. Please refer to the following document for reference:

```
Generic Inverted Index on the GPU, Technical Report (TR 11/15), School of Computing, National University of Singapore. 
```


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

To install the libGPUGenie, directlly call the install script, `install.sh`.


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




