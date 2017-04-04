#ifndef __DISTGENIE_GLOBAL_H__
#define __DISTGENIE_GLOBAL_H__

#include <vector>
#include <utility>

#ifdef NO_EXTERN
#define EXTERN
#else
#define EXTERN extern
#endif

EXTERN int g_mpi_rank, g_mpi_size;

#define MPI_DEBUG "===MPI DEBUG=== "

#endif
