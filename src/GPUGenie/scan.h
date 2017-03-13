/*
 * This module contains source code provided by NVIDIA Corporation.
 */

#ifndef SCAN_H
#define SCAN_H

#include <stdlib.h>

#define GPUGENIE_SCAN_THREADBLOCK_SIZE (256)
#define GPUGENIE_SCAN_MIN_SHORT_ARRAY_SIZE (4)
#define GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE (4*GPUGENIE_SCAN_THREADBLOCK_SIZE) // 1024
#define GPUGENIE_SCAN_MIN_LARGE_ARRAY_SIZE (8*GPUGENIE_SCAN_THREADBLOCK_SIZE) // 2048
#define GPUGENIE_SCAN_MAX_LARGE_ARRAY_SIZE (4*GPUGENIE_SCAN_THREADBLOCK_SIZE*GPUGENIE_SCAN_THREADBLOCK_SIZE) // 262144

namespace GPUGenie
{

extern const unsigned int SCAN_THREADBLOCK_SIZE;
extern const unsigned int SCAN_MIN_SHORT_ARRAY_SIZE;
extern const unsigned int SCAN_MAX_SHORT_ARRAY_SIZE;
extern const unsigned int SCAN_MIN_LARGE_ARRAY_SIZE;
extern const unsigned int SCAN_MAX_LARGE_ARRAY_SIZE;

void initScan(void);
void closeScan(void);

extern __device__ uint d_pow2ceil_32(uint x);
uint h_pow2ceil_32(uint x);

extern __global__ void g_scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size);

extern __device__ void d_scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size);

extern __global__ void g_scanInclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size);

extern __device__ void d_scanInclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size);

size_t scanExclusiveShort(
    unsigned int *d_Dst,
    unsigned int *d_Src,
    unsigned int arrayLength);

size_t scanExclusiveLarge(
    unsigned int *d_Dst,
    unsigned int *d_Src,
    unsigned int arrayLength);

void scanExclusiveHost(
    unsigned int *dst,
    unsigned int *src,
    unsigned int arrayLength);

}

#endif
