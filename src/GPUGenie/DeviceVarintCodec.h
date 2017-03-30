#ifndef DEVICE_VARINT_CODEC_H_
#define DEVICE_VARINT_CODEC_H_


#include "DeviceCompositeCodec.h"

namespace GPUGenie {

// threadblock size is 256, same for all codecs (can be up to 1024 for compute capability >= 2.0)
#define GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE (256)

// number of integers decoded by a single thread
#define GPUGENIE_CODEC_VARINT_THREAD_LOAD (4)

// maximum uncompressed length -- read from the first uint32_t of compressed word
#define GPUGENIE_CODEC_VARINT_MAX_UNCOMPR_LENGTH (GPUGENIE_CODEC_VARINT_THREAD_LOAD * \
    GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE)


/**
 *  This class defines a varint integer codec (a.k.a. variable integer, variable byte,...)
 *
 *  Supports decoding from withing a CUDA kernel.
 *
 *  Based on varint compression from Daniel Lemire.
 */
class DeviceVarintCodec : public DeviceIntegerCODEC {

public:

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t /*length*/, uint32_t *out, size_t &nvalue);

    __device__ uint32_t*
    decodeArraySequential(uint32_t *d_in, const size_t /*length*/, uint32_t *d_out, size_t &nvalue);

    __device__ uint32_t*
    decodeArrayParallel(uint32_t *d_in, size_t /* comprLength */, uint32_t *d_out, size_t &capacity);

    __device__ __host__
    ~DeviceVarintCodec() {}

    std::string
    name() const { return "DeviceVarintCodec"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_minEffectiveLength() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return GPUGENIE_CODEC_VARINT_MAX_UNCOMPR_LENGTH; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return GPUGENIE_CODEC_VARINT_THREAD_LOAD; }

private:

    void
    encodeToByteArray(uint32_t *in, const size_t length, uint8_t *bout, size_t &nvalue);

    const uint8_t*
    decodeFromByteArray(const uint8_t *inbyte, const size_t length, uint32_t *out, size_t &nvalue);

    template <uint32_t i> uint8_t
    extract7bits(const uint32_t val) {
        return static_cast<uint8_t>((val >> (7 * i)) & ((1U << 7) - 1));
    }

    template <uint32_t i> uint8_t
    extract7bitsmaskless(const uint32_t val) {
        return static_cast<uint8_t>((val >> (7 * i)));
    }

    template <class T> inline bool
    needPaddingTo32Bits(const T *inbyte) {
        return (reinterpret_cast<uintptr_t>(inbyte) & 3) != 0;
    }

    __device__ int
    numIntsStartingHere(uint32_t *d_in, int idxUnpack, int comprLength);

};

}

#endif
