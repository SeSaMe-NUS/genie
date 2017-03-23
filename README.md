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

To enable dynamic threshold update (**needs CUDA-aware MPI**), use

```bash
$ cmake -DUSE_MPI=on -DUSE_DYNAMIC=on ..
```

## Running example

Examples (tests) are available in the `bin` folder of your build directory. To run MPI GENIE, use

```bash
$ mpirun -np <n> ./bin/odgenie static/online.config.json
```

The program will listen for question on port 9090. You can send queries to the program by executing

```bash
$ nc localhost 9090
```

The query format is in JSON format, for 2 queries of dimension 5, you do

```javascript
{
  "topk": 10,
  "queries": [
    [1, 2, 3, 4, 5],
    [1, 3, 5, 7, 9]
  ]
}
```

This sends query `1 2 3 4 5` and `1 3 5 7 9` with topk set to 10.

## Attaching GDB to MPI

Run MPI with ENABLE_GDB=1 environment variable:

```bash
$ mpirun -np 2 -x ENABLE_GDB=1 ./bin/odgenie ./static/online.config.json
```

If there is only one batch of MPI processes running, we can find PID automatically. Note that we need to set variable i to non-zero value to start the process after we have attached all gdbs we want.

```bash
$ pid=$(pgrep odgenie | sed -n 1p); gdb -q --pid "${pid}" -ex "up 100" -ex "down 1" -ex "set variable gdb_attached=1" -ex "continue"
```

To attach other processes, use their corresponding PID (PID of rank 0 process + rank). Do this before starting the rank 0 process by setting variable i.

```bash
$ pid=$(pgrep odgenie | sed -n 2p); gdb -q --pid "${pid}"
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
