#ifndef __DISTGENIE_GLOBAL_H__
#define __DISTGENIE_GLOBAL_H__

#include <mutex>

#ifdef NO_EXTERN
#define EXTERN
#else
#define EXTERN extern
#endif

EXTERN int g_mpi_rank, g_mpi_size;
EXTERN std::mutex query_mutex;

const size_t BUFFER_SIZE = 10u << 20;

#define MPI_DEBUG "===MPI DEBUG=== "

#endif
