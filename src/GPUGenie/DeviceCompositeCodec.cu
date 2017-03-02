#include "scan.h"

#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"

using namespace GPUGenie;

// Explicit template instances for working codecs

template __global__ void
GPUGenie::decodeArrayParallel<DeviceJustCopyCodec>(uint32_t*, size_t, uint32_t*, size_t, size_t*);

template __global__ void
GPUGenie::decodeArrayParallel<DeviceDeltaCodec>(uint32_t*, size_t, uint32_t*, size_t, size_t*);

template __global__ void
GPUGenie::decodeArrayParallel<DeviceBitPackingCodec>(uint32_t*, size_t, uint32_t*, size_t, size_t*);

template <class CODEC> __global__ void
GPUGenie::decodeArrayParallel(uint32_t *d_Input, size_t arrayLength, uint32_t *d_Output, size_t capacity, size_t *decomprLength)
{
    CODEC codec;
    assert(blockDim.x == codec.decodeArrayParallel_lengthPerBlock() / codec.decodeArrayParallel_threadLoad());
    assert(gridDim.x <= codec.decodeArrayParallel_maxBlocks());
    assert(capacity <= gridDim.x * blockDim.x * codec.decodeArrayParallel_threadLoad());

    __shared__ uint32_t s_Input[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];
    __shared__ uint32_t s_Output[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    s_Input[idx] = (idx < (int)arrayLength) ? d_Input[idx] : 0;
    s_Output[idx] = 0;
    __syncthreads();

    codec.decodeArrayParallel(s_Input, arrayLength, s_Output, capacity);
    __syncthreads();

    if (idx < (int)capacity)
        d_Output[idx] = s_Output[idx];

    if (idx == 0 && decomprLength != NULL)
            (*decomprLength) = capacity;
}



