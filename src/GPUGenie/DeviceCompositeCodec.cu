#include "scan.h"

#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"

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

template void
GPUGenie::decodeArrayParallel<DeviceVarintCodec>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template void
GPUGenie::decodeArrayParallel<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceJustCopyCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template class
GPUGenie::DeviceCompositeCodec<DeviceBitPackingCodec,DeviceJustCopyCodec>;

template void
GPUGenie::decodeArrayParallel<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);

template class
GPUGenie::DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>;

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
    for (int i = 0; i < codec.decodeArrayParallel_lengthPerBlock(); i += codec.decodeArrayParallel_threadsPerBlock())
    {
        s_Input[idx + i] = (idx + i < (int)arrayLength) ? d_Input[idx + i] : 0;
        s_Output[idx + i] = 0;
    }

    __syncthreads();
    codec.decodeArrayParallel(s_Input, arrayLength, s_Output, capacity);
    __syncthreads();

    for (int i = 0; i < codec.decodeArrayParallel_lengthPerBlock(); i += codec.decodeArrayParallel_threadsPerBlock())
    {
        d_Output[idx + i] = s_Output[idx + i];
    }

    if (idx == 0 && d_decomprLength != NULL)
            (*d_decomprLength) = capacity;
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

    assert(length > 0 && length <= GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE);
    uint pow2arrayLength = GPUGenie::d_pow2ceil_32(length);
    uint arrayLength = (length + 3) / 4;

    // Check supported size range
    // Check parallel model compatibility
    assert(blockDim.x == GPUGENIE_SCAN_THREADBLOCK_SIZE && gridDim.x == 1);

    __syncthreads();
    GPUGenie::d_scanInclusiveShared((uint4 *)d_out, (uint4 *)d_in, arrayLength, pow2arrayLength);
    __syncthreads();
    
    nvalue = length;
    return d_in + length;
}

template <class Codec1, class Codec2> void
GPUGenie::DeviceCompositeCodec<Codec1,Codec2>::encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
    assert(length > 0);
    assert(nvalue > 0);
    int codec1minEffLength = codec1.decodeArrayParallel_minEffectiveLength();
    size_t codec1Length = (length / codec1minEffLength) * codec1minEffLength;
    size_t codec2Length = length - codec1Length;
    assert (codec1Length + codec2Length == length);
    assert (codec2Length <= length);

    size_t nvalue1 = 0;
    if (codec1Length){
        nvalue1 = nvalue;
        codec1.encodeArray(in, codec1Length, out + 1, nvalue1);
        assert(nvalue >= nvalue1); // Error - compression overflow
    }

    size_t nvalue2 = 0;
    if (codec2Length) {
        nvalue2 = nvalue - nvalue1;
        codec2.encodeArray(in + codec1Length, codec2Length, out + 1 + nvalue1, nvalue2);
        assert(nvalue - nvalue1 >= nvalue2); // Error - compression overflow
    }

    out[0] = nvalue1; // store infromation about compressed length from the first codec

    nvalue = 1 + nvalue1 + nvalue2;
}

template <class Codec1, class Codec2> const uint32_t*
GPUGenie::DeviceCompositeCodec<Codec1,Codec2>::decodeArray(const uint32_t *in, const size_t comprLength, uint32_t *out, size_t &nvalue)
{
    size_t firstCodecComprLength = *in++;

    // Codec1 decompresses as much as it can
    size_t nvalue1 = 0;
    const uint32_t *inForCodec2 = in;

    if (firstCodecComprLength){
        nvalue1 = nvalue; // set capacity for codec1 to overall capacity
        inForCodec2 = codec1.decodeArray(in, firstCodecComprLength, out, nvalue1);

        if (nvalue1 > nvalue){ // Error - Codec1 does not have enough capacity
            nvalue = nvalue1; // Set nvalue to required capacity of codec1
            return in; // Return pointer to the deginning of the compressed array
        }

        if (inForCodec2 == in + comprLength - 1){ // Codec1 decompressed everything
            nvalue = nvalue1;
            return inForCodec2;
        }
    }

    assert(inForCodec2 == in + firstCodecComprLength); // Make sure codec1 returned correct d_in pointer

    // Codec2 decompresses the leftover
    size_t nvalue2 = nvalue - nvalue1; // remaining capacity
    size_t leftoverLength = comprLength - 1 - (inForCodec2 - in);
    const uint32_t *inAfterBothCodecs = codec2.decodeArray(inForCodec2, leftoverLength, out + nvalue1, nvalue2);

    if (nvalue2 > nvalue - nvalue1){ // Error - Codec2 does not have enough capacity
        nvalue = nvalue1 + nvalue2; // Set nvalue to required capacity of codec1 + codec2
        return in; // Return pointer to the deginning of the compressed array
    }

    assert(in + comprLength - 1 == inAfterBothCodecs);
    nvalue = nvalue1 + nvalue2;
    return inAfterBothCodecs;
}


template <class Codec1, class Codec2> __device__ uint32_t*
GPUGenie::DeviceCompositeCodec<Codec1,Codec2>::decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    return nullptr;
}



template <class Codec1, class Codec2> __device__ uint32_t*
GPUGenie::DeviceCompositeCodec<Codec1,Codec2>::decodeArrayParallel(
            uint32_t *d_in, size_t comprLength, uint32_t *d_out, size_t &nvalue)
{
    size_t firstCodecComprLength = *d_in++;

    // Codec1 decompresses as much as it can
    size_t nvalue1 = 0;
    uint32_t *d_inForCodec2 = d_in;

    if (firstCodecComprLength){
        nvalue1 = nvalue; // set capacity for codec1 to overall capacity

        d_inForCodec2 = codec1.decodeArrayParallel(d_in, firstCodecComprLength, d_out, nvalue1);
        __syncthreads();

        if (nvalue1 > nvalue){ // Error - Codec1 does not have enough capacity
            nvalue = nvalue1; // Set nvalue to required capacity of codec1
            return d_in; // Return pointer to the deginning of the compressed array
        }

        if (d_inForCodec2 == d_in + comprLength - 1){ // Codec1 decompressed everything
            nvalue = nvalue1;
            return d_inForCodec2;
        }
    }

    assert(d_inForCodec2 == d_in + firstCodecComprLength); // Make sure codec1 returned correct d_in pointer

    // Codec2 decompresses the leftover
    size_t nvalue2 = nvalue - nvalue1; // remaining capacity
    size_t leftoverLength = comprLength - 1 - firstCodecComprLength;
    uint32_t *d_inAfterBothCodecs = codec2.decodeArrayParallel(d_inForCodec2, leftoverLength, d_out + nvalue1, nvalue2);
    __syncthreads();

    if (nvalue2 > nvalue - nvalue1){ // Error - Codec2 does not have enough capacity
        nvalue = nvalue1 + nvalue2; // Set nvalue to required capacity of codec1 + codec2
        return d_in; // Return pointer to the deginning of the compressed array
    }

    assert(d_in + comprLength - 1 == d_inAfterBothCodecs);
    nvalue = nvalue1 + nvalue2;
    return d_inAfterBothCodecs;
}

