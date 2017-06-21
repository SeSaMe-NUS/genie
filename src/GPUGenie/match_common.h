/** \file match_common.h
 *  \brief Basic utility functions to be used in matching kernels
 *
 * This header file has no standalone translation unit. Instead, it contains inline function definitions for matching.
 * The reason is that nvcc treats __forceinline__ and __inline__ (in release mode only) functions as static = does
 * not export symbols in the library, only inlines them, similarly to C.
 *
 * Including this file allows two separate translation units to have a copy of these functions and inline them.
 */

#ifndef GENIE_MATCH_COMMON_H
#define GENIE_MATCH_COMMON_H

#include <cstdint>

namespace genie
{
namespace core
{

typedef unsigned char u8;
typedef uint32_t u32;
typedef unsigned long long u64;

typedef u64 T_HASHTABLE;
typedef u32 T_KEY;
typedef u32 T_AGE;

/*! \struct data_
 *  \brief This is the entry format of the hash table used in GPU.
 *         Will be treated as a 64-bit unsigned integer later.
 */
typedef struct data_
{
    u32 id;/*!< Index of data point */
    float aggregation;/*!< Count of data point*/
} data_t;


const size_t MATCH_THREADS_PER_BLOCK = 256;

#define OFFSETS_TABLE_16 {0u,       3949349u, 8984219u, 9805709u,\
                          7732727u, 1046459u, 9883879u, 4889399u,\
                          2914183u, 3503623u, 1734349u, 8860463u,\
                          1326319u, 1613597u, 8604269u, 9647369u}

#define NULL_AGE 0
#define MAX_AGE 16u

const u32 KEY_TYPE_BITS = 28u;
const u32 KEY_TYPE_MASK = u32(u64((1ull) << KEY_TYPE_BITS) - 1u);
const u32 ATTACH_ID_TYPE_BITS = 32u;
const u32 ATTACH_ID_TYPE_MASK = u32(u64((1ull) << ATTACH_ID_TYPE_BITS) - 1ul);
const u32 KEY_TYPE_INIT_AGE = 1u;
const u32 KEY_TYPE_NULL_AGE = 0u;


static __device__  __constant__ u32 offsets[16];

__global__
void convert_to_data(T_HASHTABLE* table, u32 size);

} // namespace core

} // namespace genie

#endif
