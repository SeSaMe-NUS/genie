# GPUGenie

GPUGenie is a Generic Inverted Index on the GPU. It builds the database from a csv file or a vector of instances. Then GPUGenie will consturct the inverted list table and transfer it to the device. GPUGenie provides a simple way to perform the similarity queries. User may define queries and their matching ranges, then directly call the matching funtion. The library will parallel process all queries and save the matching result into a device_vector. A top k search can also be simply perfromed. GPUGenie uses parallel searching to determine the top k values in a vector. It is much faster than the CPU searching algorithm. All device methods are wrapped in host methods. Developers are not required to configure the device function call. Please refer to the following documents:

```
Generic Inverted Index on the GPU, Technical Report (TR 11/15), School of Computing, NUS. 
Generic Inverted Index on the GPU, CoRR arXiv:1603.08390 at www.comp.nus.edu.sg/~atung/publication/gpugenie.pdf
```


## Compilation and development

You are required to install gcc, g++, nvcc, boost and cmake. The minimum required versions are 3.5.1 for cmake, 7.0 for CUDA, and 1.56.0 for Boost.

To compile the program, just type 

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

This creates an "out-of-source" build of GPUGenie containing both the GPUGenie library and some tests. To use Boost which is not in the system path, change `cmake` command to

```bash
$ cmake -DBOOST_ROOT=/path/to/boost ..
```

To compile with MPI support (**currently only OpenMPI is supported**), use

```bash
$ cmake -DUSE_MPI=on ..
```

## Running example

Examples (tests) are available in the `bin` folder of your build directory. To run MPI GENIE, use

```bash
$ mpirun -np <n> ./bin/odgenie static/genie.config
```

## Documentation

Documentation for GPUGenie could be generated with Doxygen. To generate manually, type

```bash
$ cd doc
$ doxygen doxy.config
```

Documentation can also be generated with CMake. After you configure CMake following steps in [Compilation and development](#compilation-and-development), just type

```bash
$ make doc
```
