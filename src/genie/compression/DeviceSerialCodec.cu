#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"

#include "DeviceCodecTemplatesImpl.hpp"

#include "DeviceSerialCodec.h"

using namespace GPUGenie;

// Explicit template instances for Serial Codecs

template class
GPUGenie::DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>;
template class
GPUGenie::DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>;

// Explicit template instances for decoding wrapper function for Serial Codecs
// NOTE: This is intentionally separated into mutliple codec implementation files in order to facilitiate separate
// compilation units, as opposed to defining all these templates in one place
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);
template void
GPUGenie::decodeArrayParallel<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>>(int, int, uint32_t*, size_t, uint32_t*, size_t, size_t*);


template <class Codec1, class Codec2> void
GPUGenie::DeviceSerialCodec<Codec1,Codec2>::encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
    assert(length > 0);
    assert(nvalue > 0);

    uint32_t mid[nvalue];

    size_t nvalue1 = nvalue;
    codec1.encodeArray(in, length, mid, nvalue1);
    assert(nvalue1 <= nvalue); // Error - compression overflow

    size_t nvalue2 = nvalue;
    codec2.encodeArray(mid, nvalue1, out, nvalue2);
    assert(nvalue2 <= nvalue); // Error - compression overflow

    nvalue = nvalue2;
}


template <class Codec1, class Codec2> const uint32_t*
GPUGenie::DeviceSerialCodec<Codec1,Codec2>::decodeArray(const uint32_t *in, const size_t comprLength, uint32_t *out, size_t &nvalue)
{

    uint32_t mid[nvalue];

    size_t nvalue2 = nvalue;
    codec2.decodeArray(in, comprLength, mid, nvalue2);
    assert(nvalue2 <= nvalue); // Error - compression overflow
    // if (nvalue2 > nvalue){ // Error - Codec2 does not have enough capacity
    //     nvalue = nvalue2; // Set nvalue to required capacity of codec1
    //     return in; // Return pointer to the deginning of the compressed array
    // }
    
    size_t nvalue1 = nvalue; 
    codec1.decodeArray(mid, nvalue2, out, nvalue1);
    assert(nvalue1 <= nvalue); // Error - compression overflow
    // if (nvalue1 > nvalue){ // Error - Codec1 does not have enough capacity
    //     nvalue = nvalue1; // Set nvalue to required capacity of codec1
    //     return in; // Return pointer to the deginning of the compressed array
    // }

    nvalue = nvalue1;
    return out + nvalue2;
}


template <class Codec1, class Codec2> __device__ uint32_t*
GPUGenie::DeviceSerialCodec<Codec1,Codec2>::decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
{
    return nullptr;
}


template <class Codec1, class Codec2> __device__ uint32_t*
GPUGenie::DeviceSerialCodec<Codec1,Codec2>::decodeArrayParallel(
            uint32_t *d_in, size_t comprLength, uint32_t *d_out, size_t &nvalue)
{
    __shared__ uint32_t s_Mid[GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH];

    // Codec2 decompresses as much as it can
    size_t nvalue2 = nvalue; // set capacity for codec2 to overall capacity
    uint32_t *d_inAfterCodec2 = codec2.decodeArrayParallel(d_in, comprLength, s_Mid, nvalue2);
    __syncthreads();
    // Hard capacity verification
    assert(nvalue2 <= nvalue); // Error - compression overflow
    assert(d_inAfterCodec2 == d_in + comprLength);

    // // Soft capacity verification
    // if (nvalue2 > nvalue){ // Error - Codec2 does not have enough capacity
    //     nvalue = nvalue2; // Set nvalue to required capacity of codec2
    //     return d_in; // Return pointer to the beginning of the compressed array
    // }

    // Codec1 decompresses the leftover
    size_t nvalue1 = nvalue; // remaining capacity
    uint32_t *s_MidAfterCodec1 = codec1.decodeArrayParallel(s_Mid, nvalue2, d_out, nvalue1);
    __syncthreads();
    // Hard capacity verification
    assert(nvalue1 <= nvalue); // Error - compression overflow
    assert(s_MidAfterCodec1 == s_Mid + nvalue2);

    // // Soft capacity verification
    // if (nvalue1 > nvalue){ // Error - Codec1 does not have enough capacity
    //     nvalue = nvalue1; // Set nvalue to required capacity of codec2 + codec1
    //     return d_in; // Return pointer to the beginning of the compressed array
    // }

    nvalue = nvalue1; // set nvalue to final uncompressed length
    return d_inAfterCodec2;
}

