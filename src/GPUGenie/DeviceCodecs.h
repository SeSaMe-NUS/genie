#ifndef DEVICE_CODECS_H_
#define DEVICE_CODECS_H_

#include <cassert>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace GPUGenie {

template <class CODEC> __global__ void
g_decodeArrayParallel(
    uint32_t *d_Input,
    size_t arrayLength,
    uint32_t *d_Output,
    size_t capacity,
    size_t *d_decomprLength);


template <class CODEC> void
decodeArrayParallel(
        int blocks,
        int threads,
        uint32_t *d_Input,
        size_t arrayLength,
        uint32_t *d_Output,
        size_t capacity,
        size_t *d_decomprLength);


class DeviceIntegerCODEC {
public:
    __device__ __host__
    DeviceIntegerCODEC() {}

    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) = 0;

    virtual const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) = 0;

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

    virtual __device__ __host__
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

    virtual std::string
    name() const {return std::string("DeviceIntegerCODEC");};

    /** Minimal effective lenth of the compression **/
    virtual __device__ __host__ int
    decodeArrayParallel_minEffectiveLength() { return -1;}

    /** The amount of CUDA blocks this codec is able to operate on at the same time **/
    virtual __device__ __host__ int
    decodeArrayParallel_maxBlocks() {return -1;}

    /** Maximal uncompressed (or compressed) size of the array the codec is able to process **/
    virtual __device__ __host__ int
    decodeArrayParallel_lengthPerBlock() {return -1;}

    /** Maximal number of threads per single block **/
    virtual __device__ __host__ int
    decodeArrayParallel_threadsPerBlock() { return -1; }

    /** Number of decompressed values extracted by a single thread **/
    virtual __device__ __host__ int
    decodeArrayParallel_threadLoad() {return -1;}
};

class DeviceCopyMultiblockCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceCopyMultiblockCodec() : DeviceIntegerCODEC() {}

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

    __device__ uint32_t*
    decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
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
    decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
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
    ~DeviceCopyMultiblockCodec() {}

    std::string
    name() const { return "CopyMB"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 65535; }
    __device__ __host__ int decodeArrayParallel_minEffectiveLength() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 1; }

};

class DeviceCopyCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceCopyCodec() : DeviceIntegerCODEC() {}

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

    __device__ uint32_t*
    decodeArraySequential(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
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
    decodeArrayParallel(uint32_t *d_in, size_t length, uint32_t *d_out, size_t &nvalue)
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

    __device__ __host__
    ~DeviceCopyCodec() {}

    std::string
    name() const { return "Copy"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_minEffectiveLength() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 256; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }

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
    name() const { return "Delta"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_minEffectiveLength() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 256; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }
};

}

#endif
