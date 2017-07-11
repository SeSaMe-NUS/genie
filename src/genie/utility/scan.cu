/*
 * This module contains source code provided by NVIDIA Corporation.
 */

#include <assert.h>
#include <genie/utility/cuda_macros.h>

#include "scan.h"

const uint THREADBLOCK_SIZE = GPUGENIE_SCAN_THREADBLOCK_SIZE;

const uint genie::utility::SCAN_THREADBLOCK_SIZE     = GPUGENIE_SCAN_THREADBLOCK_SIZE;
const uint genie::utility::SCAN_MIN_SHORT_ARRAY_SIZE = GPUGENIE_SCAN_MIN_SHORT_ARRAY_SIZE;
const uint genie::utility::SCAN_MAX_SHORT_ARRAY_SIZE = GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE;
const uint genie::utility::SCAN_MIN_LARGE_ARRAY_SIZE = GPUGENIE_SCAN_MIN_LARGE_ARRAY_SIZE;
const uint genie::utility::SCAN_MAX_LARGE_ARRAY_SIZE = GPUGENIE_SCAN_MAX_LARGE_ARRAY_SIZE;

// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half with 'size' zeros avoiding if(pos >= offset) condition
// evaluation and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}


inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

// Exclusive vector scan: the array to be scanned is stored in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void genie::utility::g_scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

__device__ void genie::utility::d_scanExclusivePerBlockShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

__device__ void genie::utility::d_scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

__global__ void genie::utility::g_scanInclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Inclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

__device__ void genie::utility::d_scanInclusivePerBlockShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Inclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

__device__ void genie::utility::d_scanInclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint activeThreads,
    uint pow2size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = (pos < activeThreads) ? d_Src[pos] : uint4{0,0,0,0};

    //Calculate exclusive scan
    uint4 odata4 = scan4Inclusive(idata4, s_Data, pow2size);

    //Write back
    if (pos < activeThreads)
        d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(
    uint *d_Buf,
    uint *d_Dst,
    uint *d_Src,
    uint arrayLength,
    uint blocks)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= blocks)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;

    uint sumLocation;
    if (pos < blocks - 1)
        sumLocation = (4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos;
    else
        sumLocation = arrayLength;

    idata = 
        d_Dst[sumLocation] + d_Src[sumLocation];

    //Compute
    uint odata = scan1Exclusive(idata, s_Data, blocks);

    //Avoid out-of-bound access
    if (pos < blocks)
    {
        d_Buf[pos] = odata;
    }
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(
    uint4 *d_Data,
    uint *d_Buffer,
    uint arrayLength)
{
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        buf = d_Buffer[blockIdx.x];
    }

    __syncthreads();

    if (pos < arrayLength)
    {   
        uint4 data4 = d_Data[pos];
        data4.x += buf;
        data4.y += buf;
        data4.z += buf;
        data4.w += buf;
        d_Data[pos] = data4;
    }
}


//Internal exclusive scan buffer
static uint *d_Buf;

void genie::utility::initScan(void)
{
    cudaCheckErrors(cudaMalloc((void **)&d_Buf, THREADBLOCK_SIZE * sizeof(uint)));
}

void genie::utility::closeScan(void)
{
    cudaCheckErrors(cudaFree(d_Buf));
}


// Returns the first power of two greater or equal to x
__device__ uint genie::utility::d_pow2ceil_32 (uint x)
{
    if (x == 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

// Returns the first power of two greater or equal to x
uint genie::utility::h_pow2ceil_32 (uint x)
{
    if (x == 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

size_t genie::utility::scanExclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint arrayLength)
{
    // Check the array length is a mutliple of 4. This is because we use uint4 processed by a single thread.
    assert(arrayLength % 4 == 0);

    //Check power-of-two factorization
    uint pow2arrayLength = h_pow2ceil_32(arrayLength);
    // printf("power of two size: %u\n", pow2arrayLength);
    assert(pow2arrayLength >= arrayLength);

    // Check supported size range
    assert((pow2arrayLength >= SCAN_MIN_SHORT_ARRAY_SIZE) && (pow2arrayLength <= SCAN_MAX_SHORT_ARRAY_SIZE));

    // printf("running scanExclusiveShort on %d blocks each of %d threads, total active threads: %d\n",
    //     (pow2arrayLength+(4*THREADBLOCK_SIZE)-1)/(4*THREADBLOCK_SIZE),THREADBLOCK_SIZE, arrayLength/4);

    g_scanExclusiveShared<<<(pow2arrayLength+(4*THREADBLOCK_SIZE)-1)/(4*THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        arrayLength / 4,
        pow2arrayLength
    );
    CUDA_LAST_ERROR();

    return THREADBLOCK_SIZE;
}

size_t genie::utility::scanExclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint arrayLength)
{
    // Check the array length is a mutliple of 4. This is because we use uint4 processed by a single thread.
    assert(arrayLength % 4 == 0);

    //Check power-of-two factorization
    uint pow2arrayLength = h_pow2ceil_32(arrayLength);
    // printf("power of two size: %u\n", pow2arrayLength);
    assert(pow2arrayLength >= (arrayLength));

    //Check supported size range
    assert((pow2arrayLength >= SCAN_MIN_LARGE_ARRAY_SIZE) && (pow2arrayLength <= SCAN_MAX_LARGE_ARRAY_SIZE));

    // printf("running scanExclusiveLong on %d blocks each of %d threads\n",
        // (pow2arrayLength + (4 * THREADBLOCK_SIZE) - 1) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);

    g_scanExclusiveShared<<<(pow2arrayLength + (4 * THREADBLOCK_SIZE) - 1) / (4 * THREADBLOCK_SIZE),
                           THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        arrayLength / 4,
        4 * THREADBLOCK_SIZE
    );
    CUDA_LAST_ERROR();

    //Not all threadblocks need to be packed with input data:
    //inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 = iDivUp(pow2arrayLength / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
    scanExclusiveShared2<<< blockCount2, THREADBLOCK_SIZE>>>(
        (uint *)d_Buf,
        (uint *)d_Dst,
        (uint *)d_Src,
        arrayLength, // uses the original arrayLength for uint array, unlike scanExclusiveShared and uniformUpdate
        pow2arrayLength / (4 * THREADBLOCK_SIZE)
    );
    CUDA_LAST_ERROR();

    uniformUpdate<<<pow2arrayLength / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint *)d_Buf,
        (arrayLength + 3) / 4
    );
    CUDA_LAST_ERROR();

    return THREADBLOCK_SIZE;
}

void genie::utility::scanExclusiveHost(
    uint *dst,
    uint *src,
    uint arrayLength)
{
    dst[0] = 0;

    for (uint j = 1; j < arrayLength; j++)
        dst[j] = src[j - 1] + dst[j - 1];
}



