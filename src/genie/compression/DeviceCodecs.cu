#include <genie/utility/scan.h>

// This includes the implementation of g_decodeArrayParallel and decodeArrayParallel wrapper. Every implementation file
// that includes templates definictions of these functions needs to include their implementation as well
#include "DeviceDeltaHelper.h"

#include "DeviceCodecs.h"

#include "DeviceCodecTemplatesImpl.hpp"

using namespace GPUGenie;

// Explicit template instances for CPU decoding wrapper function of simple codecs
// NOTE: This is intentionally separated into mutliple codec implementation files in order to facilitiate separate
// compilation units, as opposed to defining all these templates in one place.
template void
GPUGenie::decodeArrayParallel<DeviceCopyCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceCopyMultiblockCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceDeltaCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);


__device__ uint32_t*
GPUGenie::DeviceCopyMultiblockCodec::decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    if (length > nvalue){
        // We do not have enough capacity in the decompressed array!
        nvalue = length;
        return d_in;
    }
    for (int i = 0; i < (int)length; i++)
        d_out[i] = d_in[i];
    nvalue = length;
    return d_in + length;
}

__device__ uint32_t*
GPUGenie::DeviceCopyMultiblockCodec::decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    assert(length <= gridDim.x * blockDim.x); // 1 thread copies one value
    assert(length <= nvalue); // not enough capacity in the decompressed array!

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length)
        d_out[idx] = d_in[idx];
    __syncthreads();

    nvalue = length;
    return d_in + length;
}

__device__ uint32_t*
GPUGenie::DeviceCopyCodec::decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    if (length > nvalue){
        // We do not have enough capacity in the decompressed array!
        nvalue = length;
        return d_in;
    }
    for (int i = 0; i < length; i++)
        d_out[i] = d_in[i];
    nvalue = length;
    return d_in + length;
}


__device__ uint32_t*
GPUGenie::DeviceCopyCodec::decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    assert(length <= decodeArrayParallel_lengthPerBlock());
    assert(length <= nvalue); // not enough capacity in the decompressed array!

    int idx = threadIdx.x;
    int fullThreadBlockLimit = length - decodeArrayParallel_threadsPerBlock();
    int i = 0;
    for (; i <= fullThreadBlockLimit; i += decodeArrayParallel_threadsPerBlock())
    {
        d_out[idx + i] = d_in[idx + i];
    }
    if (idx + i < length)
        d_out[idx + i] = d_in[idx + i];
    __syncthreads();

    nvalue = length;
    return d_in + length;
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
    assert(length <= nvalue); // not enough capacity in the decompressed array!
    assert(length > 0 && length <= decodeArrayParallel_lengthPerBlock());
    uint pow2arrayLength = GPUGenie::d_pow2ceil_32(length);
    uint arrayLength = (length + 3) / 4;

    // Check supported size range
    // Check parallel model compatibility
    assert(blockDim.x == GPUGENIE_SCAN_THREADBLOCK_SIZE);

    __syncthreads();
    GPUGenie::d_scanInclusivePerBlockShared((uint4 *)d_out, (uint4 *)d_in, arrayLength, pow2arrayLength);
    __syncthreads();
    
    nvalue = length;
    return d_in + length;
}


