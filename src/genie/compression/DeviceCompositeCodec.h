#ifndef DEVICE_COMPOSITE_CODEC_H_
#define DEVICE_COMPOSITE_CODEC_H_

#include <sstream>
#include <string>
#include <vector>

#include "DeviceCodecs.h"

namespace genie
{
namespace compression
{

/**
 * This is a useful class for CODEC that only compress data having length a multiple of some unit length.
 */
template <class Codec1, class Codec2>
class DeviceCompositeCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceCompositeCodec() : codec1(), codec2() {
        assert(codec1.decodeArrayParallel_minEffectiveLength() > 1);
        assert(codec2.decodeArrayParallel_minEffectiveLength() == 1);
    }

    Codec1 codec1;
    Codec2 codec2;

    __device__ __host__
    ~DeviceCompositeCodec() {}

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
        convert << "Composite(" << codec1.name() << "-" << codec2.name() << ")";
        return convert.str();
    }

    __device__ __host__ int
    decodeArrayParallel_minEffectiveLength() { 
        return codec2.decodeArrayParallel_minEffectiveLength();
    }

    __device__ __host__ int
    decodeArrayParallel_maxBlocks() { 
        return min(codec1.decodeArrayParallel_maxBlocks(), codec2.decodeArrayParallel_maxBlocks());
    }

    __device__ __host__ int
    decodeArrayParallel_lengthPerBlock() {
        assert(codec1.decodeArrayParallel_lengthPerBlock() == codec2.decodeArrayParallel_lengthPerBlock());
        return codec1.decodeArrayParallel_lengthPerBlock();
    }

    __device__ __host__ int
    decodeArrayParallel_threadsPerBlock() {
        return min(codec1.decodeArrayParallel_threadsPerBlock(),codec2.decodeArrayParallel_threadsPerBlock());
    }

    __device__ __host__ int
    decodeArrayParallel_threadLoad() {
        return max(codec1.decodeArrayParallel_threadLoad(),codec2.decodeArrayParallel_threadLoad());
    }
};

} // namespace compression
} // namespace genie

#endif
