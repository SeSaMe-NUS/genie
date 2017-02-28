#include "scan.h"

#include "DeviceCompositeCodec.h"

using namespace GPUGenie;

template __global__ void GPUGenie::decodeArrayParallel<DeviceJustCopyCodec>(uint32_t*, uint32_t*, size_t, size_t);
template __global__ void GPUGenie::decodeArrayParallel<DeviceDeltaCodec>(uint32_t*, uint32_t*, size_t, size_t);

template <class CODEC> __global__ void
GPUGenie::decodeArrayParallel(uint32_t *d_Input, uint32_t *d_Output, size_t arrayLength, size_t &capacity)
{
    assert((arrayLength >= GPUGENIE_SCAN_MIN_SHORT_ARRAY_SIZE) && (arrayLength <= GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE));
    assert(capacity <= GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE);
    assert(blockDim.x == GPUGENIE_SCAN_THREADBLOCK_SIZE);
    assert(gridDim.x == 1);

    const size_t s = GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE;
    __shared__ uint32_t s_Input[s];
    __shared__ uint32_t s_Output[s];

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    s_Input[idx] = (idx < arrayLength) ? d_Input[idx] : 0;
    s_Output[idx] = 0;
    __syncthreads();

    CODEC codec;
    codec.decodeArrayParallel(s_Input, arrayLength, s_Output, capacity);

    if (idx == 0)
    {
        printf("Codec used %d of %d s_Output array.\n", capacity, GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE);
    }
}

