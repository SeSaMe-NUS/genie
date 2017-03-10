#include "scan.h"

#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"

using namespace GPUGenie;

// Explicit template instances for working codecs

template void
GPUGenie::decodeArrayParallel<DeviceJustCopyCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template void
GPUGenie::decodeArrayParallel<DeviceDeltaCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template void
GPUGenie::decodeArrayParallel<DeviceBitPackingCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template void
GPUGenie::decodeArrayParallel<DeviceBitPackingPrefixedCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);


template <class CODEC> void
GPUGenie::decodeArrayParallel(
        int blocks,
        int threads,
        uint32_t *d_Input,
        size_t arrayLength,
        uint32_t *d_Output,
        size_t capacity,
        size_t *d_decomprLength)
{
    g_decodeArrayParallel<CODEC><<<blocks,threads>>>(d_Input, arrayLength, d_Output, capacity, d_decomprLength);
}

template <class CODEC> __global__ void
GPUGenie::g_decodeArrayParallel(uint32_t *d_Input, size_t arrayLength, uint32_t *d_Output, size_t capacity, size_t *d_decomprLength)
{
    CODEC codec;
    assert(blockDim.x == codec.decodeArrayParallel_lengthPerBlock() / codec.decodeArrayParallel_threadLoad());
    assert(gridDim.x <= codec.decodeArrayParallel_maxBlocks());
    assert(capacity <= gridDim.x * blockDim.x * codec.decodeArrayParallel_threadLoad());

    
    __shared__ uint32_t s_Input[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];
    __shared__ uint32_t s_Output[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("d_Input[%d] = 0x%08X \n", idx, d_Input[idx]);

    s_Input[idx] = (idx < (int)arrayLength) ? d_Input[idx] : 0;
    s_Output[idx] = 0;

    printf("s_Input[%d] = 0x%08X \n", idx, s_Input[idx]);

    __syncthreads();
    codec.decodeArrayParallel(s_Input, arrayLength, s_Output, capacity);
    __syncthreads();

    printf("s_Output[%d] = 0x%08X \n", idx, s_Output[idx]);

    if (idx < (int)capacity)
        d_Output[idx] = s_Output[idx];

    if (idx == 0 && d_decomprLength != NULL)
            (*d_decomprLength) = capacity;

    printf("d_Output[%d] = 0x%08X \n", idx, s_Output[idx]);
}


void
GPUGenie::DeviceDeltaCodec::encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
    std::memcpy(out, in, sizeof(uint32_t) * length);
    DeviceDeltaHelper<uint32_t>::delta(0, out, length);
    nvalue = length;
}

const uint32_t*
GPUGenie::DeviceDeltaCodec::decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
    std::memcpy(out, in, sizeof(uint32_t) * length);
    DeviceDeltaHelper<uint32_t>::inverseDelta(0, out, length);
    nvalue = length;
    return in + length;
}

__device__ const uint32_t*
GPUGenie::DeviceDeltaCodec::decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
{
    if (length > nvalue){
        // We do not have enough capacity in the decompressed array!
        nvalue = length;
        return d_in;
    }
    for (int i = 0; i < length; i++)
        d_out[i] = d_in[i];
    DeviceDeltaHelper<uint32_t>::inverseDeltaOnGPU(0, d_out, length);
    nvalue = length;
    return d_in + length;
}

__device__ uint32_t*
GPUGenie::DeviceDeltaCodec::decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    assert(length <= gridDim.x * blockDim.x * 4); // one thread can process 4 values
    assert(length <= nvalue); // not enough capacity in the decompressed array!
    assert(blockIdx.x == 0); // currently only support single block

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[idx] = 0; // d_out should be shared memory

    assert(length > 0 && length <= GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE);
    uint pow2arrayLength = GPUGenie::d_pow2ceil_32(length);
    uint arrayLength = (length + 3) / 4;


    // Check supported size range
    // Check parallel model compatibility
    assert(blockDim.x == GPUGENIE_SCAN_THREADBLOCK_SIZE && gridDim.x == 1);

    __syncthreads();
    GPUGenie::d_scanExclusiveShared((uint4 *)d_out, (uint4 *)d_in, arrayLength, pow2arrayLength);
    __syncthreads();
    
    if (idx == 0)
        assert(d_out[idx] == 0);
    else if (idx < arrayLength)
        assert(d_out[idx] >= d_out[idx-1]);

    // turn it into inclusive scan
    uint32_t inc = 0;
    if (idx < length)
        inc = d_out[idx+1];
    __syncthreads();
    if (idx < length)
        d_out[idx] = inc;

    nvalue = length;
    return d_in + length;
}

