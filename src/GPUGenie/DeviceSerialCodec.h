#ifndef DEVICE_SERIAL_CODEC_H_
#define DEVICE_SERIAL_CODEC_H_

#include <algorithm>
#include <string>

#include "DeviceCompositeCodec.h"


// threadblock size is 256, same for all codecs (can be up to 1024 for compute capability >= 2.0)
#define GPUGENIE_CODEC_SERIAL_THREADBLOCK_SIZE (256)

// number of integers decoded by a single thread
#define GPUGENIE_CODEC_SERIAL_THREAD_LOAD (4)

// maximum uncompressed length -- read from the first uint32_t of compressed word
#define GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH (GPUGENIE_CODEC_VARINT_THREAD_LOAD * \
    GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE)



namespace GPUGenie {

template <class Codec1, class Codec2>
class DeviceSerialCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceSerialCodec() : codec1(), codec2() {
        // Check both codecs have the same parallel model for decompression on GPU
        assert(codec1.decodeArrayParallel_lengthPerBlock() == codec2.decodeArrayParallel_lengthPerBlock());
        assert(codec1.decodeArrayParallel_threadsPerBlock() == codec2.decodeArrayParallel_threadsPerBlock());
        assert(codec1.decodeArrayParallel_threadLoad() == codec2.decodeArrayParallel_threadLoad());
        assert(codec1.decodeArrayParallel_maxBlocks() == codec2.decodeArrayParallel_maxBlocks());
    }

    Codec1 codec1;
    Codec2 codec2;

    __device__ __host__
    ~DeviceSerialCodec() {}

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    __device__ uint32_t*
    decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue);

    __device__ uint32_t*
    decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue);

    std::string
    name() const {
        std::ostringstream convert;
        convert << "Serial<" << codec1.name() << "," << codec2.name() << ">";
        return convert.str();
    }

    __device__ __host__ int
    decodeArrayParallel_minEffectiveLength() { 
        return max(codec1.decodeArrayParallel_minEffectiveLength(),
            codec2.decodeArrayParallel_minEffectiveLength());
    }

    __device__ __host__ int
    decodeArrayParallel_maxBlocks() { 
        return codec1.decodeArrayParallel_maxBlocks();
    }

    __device__ __host__ int
    decodeArrayParallel_lengthPerBlock() {
        return codec1.decodeArrayParallel_lengthPerBlock();
    }

    __device__ __host__ int
    decodeArrayParallel_threadsPerBlock() {
        return codec1.decodeArrayParallel_threadsPerBlock();
    }

    __device__ __host__ int
    decodeArrayParallel_threadLoad() {
        return codec1.decodeArrayParallel_threadLoad();
    }
};

}

#endif
