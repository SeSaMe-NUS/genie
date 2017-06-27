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
struct data_t
{
    uint32_t id;/*!< Index of data point */
    float aggregation;/*!< Count of data point*/
};


const size_t MATCH_THREADS_PER_BLOCK = 256;

__global__
void convert_to_data(T_HASHTABLE* table, u32 size);

} // namespace core

} // namespace genie

#endif
