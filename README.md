# GENIE

GENIE is a Generic Inverted Index on the GPU. It builds the database from a csv file or a vector of instances. Then
GENIE will consturct the inverted list table and transfer it to the device. GENIE provides a simple way to
perform the similarity queries. User may define queries and their matching ranges, then directly call the matching
funtion. The library will parallel process all queries and save the matching result into a device_vector. A top k
search can also be simply perfromed. GENIE uses parallel searching to determine the top k values in a vector. It is
much faster than the CPU searching algorithm. All device methods are wrapped in host methods. Developers are not
required to configure the device function call. Please refer to the following documents:

```
Generic Inverted Index on the GPU, Technical Report (TR 11/15), School of Computing, NUS. 
Generic Inverted Index on the GPU, CoRR arXiv:1603.08390 at www.comp.nus.edu.sg/~atung/publication/gpugenie.pdf
```


## Compilation and development

You are required to install G++, CMake, CUDA, OpenMPI and Boost. The minimum required versions are:
- GCC with C++11 support (4.8)
- CMake 3.8
- CUDA 7.0
- OpenMPI 1.7 (for `GENIE_DISTRIBUTED` only)
- Boost 1.63: serialization, iostreams, program_options (for `GENIE_COMPR` only)

To create an "out-of-source" build of GENIE containing both the GENIE library, tests and tools, you can use the
standard CMake procedure:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
```
Use target `$ make test` to run GENIE tests, `$ make doc` to build html code documentation, `$ make install` to install GENIE.

`CMake` build parameters can be further configured using the following options:
- `CMAKE_BUILD_TYPE:STRING` -- build type, one of `Release`, `Debug` (default `Debug`)
- `CMAKE_INSTALL_PREFIX:PATH` -- `cmake`'s option for installation prefix (default `${CMAKE_BINARY_DIR}/install`)
- `BOOST_ROOT:PATH` -- root dir of Boost libraries (default from system paths)
- `DOXYGEN_EXECUTABLE:PATH` -- doxygen executable (default from system paths)
- `MPI_HOME:PATH` -- root dir of OpenMPI installation (default from system paths)
- `GENIE_DISTRIBUTED:BOOL` -- enable distributed GENIE module (default OFF)
- `GENIE_COMPR:BOOL` -- enable compression GENIE module (default ON)
- `GENIE_SIMDCAI:BOOL` -- enable compilation of SIMDCAI library (default OFF)
- `GENIE_EXAMPLES:BOOL` -- enable compilation of GENIE examples (default ON)

Example use of `cmake` command may look like this:
```bash
$ cmake -DGENIE_SIMDCAI=ON -DCMAKE_BUILD_TYPE=Release -DGENIE_DISTRIBUTED=ON -DGENIE_COMPR=ON \
        -DBOOST_ROOT=/home/lubos/boost -DCMAKE_INSTALL_PREFIX=/home/lubos/genie-install ..
```

## Running GENIE

There are several main parts of GENIE project. The core is a library `/lib/libgenie.a` with the main functionality.
To see how to use the library, you can check source code in either `/example` or `/test`. Tests are the simplest
applications built on top of GENIE library. Other utilities include a compression performance toolkit in `/perf` and
miscellaneous utilities in `/utility`. All of these tools are compiled into `/bin` directory.


### Compression performance toolkit


Compression toolkit `compression` is compiled into `bin` directory. It's a standalone app for performance measurements
of GENIE focused mainly on compression capability.

To see all options of the compression performance toolkit, run:
```bash
$ ./compression --help
```


### Multi-node Multi-GPU example

The current version supports loading multiple tables (i.e. clusters) and
executing queries on corresponding clusters.

#### Preparing dataset and queries

##### Dataset

First, you need to cluster a single `.csv` file into multiple files with
one cluster of data in each file. Make sure the files have a common prefix
in their file names and that the file names end with `_<cluster-id>`
(e.g. `sift_0.csv`, `sift_1.csv`).

After the clustering, you need to further split each file into multiple
files for loading onto multiple GPUs. This could be done with the `split.sh`
script in the `utility` folder (**currently it splits into 2 files for 2 GPUs**).
If a cluster is named `sift_0.csv`, this will split it into `sift_0_0.csv`
and `sift_0_1.csv`.

Once the files are ready, you may want to convert them into binary format
for faster loading (if you want to experiment a few times, this greatly
speeds up the loading process for subsequent runs). The conversion could
be done by the `csv2binary` program in `bin`. A helper script, `convert.sh`
is also provided to convert multiple files at once. For example, given 20
clusters and 2 GPUs (with prefix `sift`), we can do

```bash
$ bash convert.sh 20 2 sift
```

To make sure everything works, please place the above mentioned programs/scripts
and your clustered `.csv` files into a single directory before processing the files.

##### Queries

The query is in JSON format, for 2 queries of dimension 5, you do

```javascript
{
  "topk": 10,
  "queries": [
    {
      "content": [1, 2, 3, 4, 5],
      "clusters": [0, 9, 13]
    },
    {
      "content": [90, 24, 33, 14, 5],
      "clusters": [3, 7]
    }
  ]
}
```

This sends query `1 2 3 4 5` and `90 24 33 14 5` with topk set to 10.
Currently the user needs to specify the clusters to search for a
given query.

#### Running MPIGenie

To run MPIGenie on a single node, use

```bash
$ mpirun -np <n> ./bin/odgenie static/online.config.json
```

To run MPIGenie on multiple nodes, use

```bash
$ /path/to/mpirun -np <n> -hostfile hosts ./bin/odgenie static/online.config.json
```

An example `hosts` file looks like

```
slave1 slots=3
slave2 slots=3
slave3 slots=3
slave4 slots=3
```

Sometimes, you may need to specify which network interface to use

```bash
$ /path/to/mpi -np <n> -hostfile hosts --mca btl_tcp_if_include <interface> ./bin/odgenie static/online.config.json
```

**Modify the configuration file accordingly.** If you converted files into
binary format, set `data_format` to be 1 in the configuration file.

The program will listen for question on port 9090. You can send queries
to the program by executing

```bash
$ nc <hostname> 9090 < queries.json
```



## Debugging


### Local Debugging

Use `cuda-gdb` the same way you would use `gdb`. Example workflow of debugging the `compression` toolset may look like
this:

``` bash
$ cd debug && cmake -DCMAKE_BUILD_TYPE=Release .. && make && cd bin
$ rm -rf log && cuda-gdb -ex "source .b" --args ./compression integrated /home/lubos/data/adult.csv all
```
where ".b" contains gdb breakpoints and cuda-gdb autosteps (using `save b .b` from gdb or cuda-gdb).


### Distributed GENIE

Run MPI with ENABLE_GDB=1 environment variable:

```bash
$ mpirun -np 2 -x ENABLE_GDB=1 ./bin/odgenie ./static/online.config.json
```

If there is only one batch of MPI processes running, we can find PID automatically. Note that we need to set variable i
to non-zero value to start the process after we have attached all gdbs we want.

```bash
$ pid=$(pgrep odgenie | sed -n 1p); gdb -q --pid "${pid}" -ex "up 100" -ex "down 1" -ex "set variable gdb_attached=1" -ex "continue"
```

To attach other processes, use their corresponding PID (PID of rank 0 process + rank). Do this before starting the rank
0 process by setting variable i.

```bash
$ pid=$(pgrep odgenie | sed -n 2p); gdb -q --pid "${pid}"
```


## Documentation

Code documentation for GENIE can be generated with `cmake` and `make`. After you configure CMake following steps in
[Compilation and Development](#compilation-and-development), just run `$ make doc`.
