#ifndef DEVICE_COMPOSITE_CODEC_H_
#define DEVICE_COMPOSITE_CODEC_H_

#include <string>
#include <vector>

#include <SIMDCAI/include/common.h>
#include <SIMDCAI/include/util.h>
#include <SIMDCAI/include/codecs.h>

#include "DeviceDeltaHelper.h"
#include "scan.h"

namespace GPUGenie {

template <class CODEC> void
decodeArrayParallel(
        int blocks,
        int threads, 
        uint32_t *d_Input,
        size_t arrayLength,
        uint32_t *d_Output,
        size_t capacity,
        size_t *decomprLength = NULL);

template <class CODEC> __global__ void
g_decodeArrayParallel(
        uint32_t *d_Input,
        size_t arrayLength,
        uint32_t *d_Output,
        size_t capacity,
        size_t *decomprLength = NULL);


class DeviceIntegerCODEC {
public:
    __device__ __host__
    DeviceIntegerCODEC() {}

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {};

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {return NULL;};

    /**
        The function must make sure not to write in d_out[nvalue] and beyond. If decompressed size is greater than
        the capacity (nvalue), there is no need to write any output at all.

        Node, this function should be pure virtual, but CUDA doesn't allow combination of virtual functions on both 
        __host__ and __device__. If a function is pure virtual, it has to be overridden on both __host__ and __device__
    **/
    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {return NULL;}

    /**
        Decompress compressed list using thread parallelism.

        Node, this function should be pure virtual, but CUDA doesn't allow combination of virtual functions on both 
        __host__ and __device__.
    **/
    __device__ const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {return NULL;}

    __device__ __host__
    ~DeviceIntegerCODEC() {}

    /** Convenience function not supported */
    std::vector<uint32_t>
    compress(std::vector<uint32_t> &data) {
        throw std::logic_error("DeviceIntegerCODEC::compress not supported!");
    }

    /** Convenience function not supported */
    std::vector<uint32_t>
    uncompress(std::vector<uint32_t> &compresseddata, size_t expected_uncompressed_size = 0) {
        throw std::logic_error("DeviceIntegerCODEC::uncompress not supported!");
    }

    std::string
    name() const {return std::string("DeviceIntegerCODEC");};

    /** The amount of CUDA blocks this codec is able to operate on at the same time **/
    __device__ __host__
    int decodeArrayParallel_maxBlocks() {return -1;}

    /** Maximal uncompressed (or compressed) size of the array the codec is able to process **/
    __device__ __host__ int
    decodeArrayParallel_lengthPerBlock() {return -1;}

    /** Maximal number of threads per single block **/
    __device__ __host__ int
    decodeArrayParallel_threadsPerBlock() { return -1; }

    /** Number of decompressed values extracted by a single thread **/
    __device__ __host__ int
    decodeArrayParallel_threadLoad() {return -1;}
};

class DeviceJustCopyCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceJustCopyCodec() : DeviceIntegerCODEC() {}

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        nvalue = length;
    }

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        nvalue = length;
        return in + length;
    }

    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
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

    /**
        This function may access d_in[(length+3)/4] and d_out[(length+3)/4] due to scan4 optimization
     */
    __device__ const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
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

    __device__ __host__
    ~DeviceJustCopyCodec() {}

    std::string
    name() const { return "DeviceJustCopyCodec"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 65535; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 1; }

};


class DeviceDeltaCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceDeltaCodec() : DeviceIntegerCODEC() {}

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue);

    __device__ uint32_t*
    decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue);

    __device__ __host__
    ~DeviceDeltaCodec() {}

    std::string
    name() const { return "DeviceDeltaCodec"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 256; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }
};



/**
 * Same as SIMDCAI::CompositeCodes, but support decoding on GPU
 *
 * This is a useful class for CODEC that only compress data having length a multiple of some unit length.
 */
template <class Codec1, class Codec2>
class DeviceCompositeCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceCompositeCodec() : codec1(), codec2() {}

    Codec1 codec1;
    Codec2 codec2;

    __device__ __host__
    ~DeviceCompositeCodec() {}

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
        const size_t roundedlength = length / Codec1::BlockSize * Codec1::BlockSize;
        size_t nvalue1 = nvalue;
        codec1.encodeArray(in, roundedlength, out, nvalue1);

        if (roundedlength < length) {
            assert(nvalue >= nvalue1);
            size_t nvalue2 = nvalue - nvalue1;
            codec2.encodeArray(in + roundedlength, length - roundedlength, out + nvalue1, nvalue2);
            nvalue = nvalue1 + nvalue2;
        } else {
            nvalue = nvalue1;
        }
    }

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
        const uint32_t *const initin(in);
        size_t mynvalue1 = nvalue;
        const uint32_t *in2 = codec1.decodeArray(in, length, out, mynvalue1);
        if (length + in > in2) {
            assert(nvalue > mynvalue1);
            size_t nvalue2 = nvalue - mynvalue1;
            const uint32_t *in3 = codec2.decodeArray(in2, length - (in2 - in), out + mynvalue1, nvalue2);
            nvalue = mynvalue1 + nvalue2;
            assert(initin + length >= in3);
            return in3;
        }
        nvalue = mynvalue1;
        assert(initin + length >= in2);
        return in2;
    }

    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {
        const uint32_t *const initin(d_in);
        size_t mynvalue1 = nvalue;
        const uint32_t *d_in2 = codec1.decodeArraySequential(d_in, length, d_out, mynvalue1);
        if (mynvalue1 > nvalue){ // Codec1 does not have enough capacity
            nvalue = mynvalue1;
            return d_in;
        }
        if (length + d_in > d_in2) {
            assert(nvalue > mynvalue1);
            size_t nvalue2 = nvalue - mynvalue1;
            const uint32_t *in3 = codec2.decodeArraySequential(d_in2, length - (d_in2 - d_in), d_out + mynvalue1, nvalue2);
            if (nvalue2 > nvalue - mynvalue1){ // Codec2 does not have enough capacity
                nvalue = mynvalue1 + nvalue2;
                return d_in;
            }
            nvalue = mynvalue1 + nvalue2;
            assert(initin + length >= in3);
            return in3;
        }
        nvalue = mynvalue1;
        assert(initin + length >= d_in2);
        return d_in2;
    }

    __device__ const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {
        return NULL;
    }

    std::string name() const {
        std::ostringstream convert;
        convert << "DeviceCompositeCodec_" << codec1.name() << "+" << codec2.name();
        return convert.str();
    }

    __device__ __host__ int
    decodeArrayParallel_maxBlocks() { 
        assert(codec1.decodeArrayParallel_maxBlocks() == codec2.decodeArrayParallel_maxBlocks());
        return codec1.decodeArrayParallel_maxBlocks();
    }

    __device__ __host__ int
    decodeArrayParallel_lengthPerBlock() {
        assert(codec1.decodeArrayParallel_lengthPerBlock() == codec2.decodeArrayParallel_lengthPerBlock());
        return codec1.decodeArrayParallel_lengthPerBlock();
    }

    __device__ __host__ int
    decodeArrayParallel_threadsPerBlock() {
        assert(codec1.decodeArrayParallel_threadsPerBlock() == codec2.decodeArrayParallel_threadsPerBlock());
        return codec1.decodeArrayParallel_threadsPerBlock();
    }

    __device__ __host__ int
    decodeArrayParallel_threadLoad() {
        assert(codec1.decodeArrayParallel_threadLoad() == codec2.decodeArrayParallel_threadLoad());
        return codec1.decodeArrayParallel_threadLoad();
    }
};

}

#endif
