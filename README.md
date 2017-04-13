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

Examples (tests) are available in the `bin` folder of your build directory.

### Multi-node Multi-GPU example

The current version support loading multiple tables (i.e. clusters) and
executing queries on corresponding clusters.

#### Preparing dataset and queries

##### Dataset

First, you need to cluster a single `.csv` file into multiple files with
one cluster of data in each file. Make sure the files have a common prefix
in their file names and that the file names ends with `_<cluster-id>`
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
clusters and 2 GPUs, we can do

```bash
$ bash convert.sh 20 2
```

**If you converted files into binary format, set `data_format` to be 1 in the
configuration file.**

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

To run MPIGenie, use

```bash
$ mpirun -np <n> ./bin/odgenie static/online.config.json
```

**Modify the configuration file accordingly.** The program will listen for question on port 9090.
You can send queries to the program by executing

```bash
$ nc localhost 9090 < queries.json
```

## Debugging

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
