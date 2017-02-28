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

template <class CODEC> __global__ void
decodeArrayParallel(uint32_t *d_Input, uint32_t *d_Output, size_t arrayLength, size_t &capacity);

class DeviceIntegerCODEC {
public:
    __device__ __host__
    DeviceIntegerCODEC() {}

    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {};

    virtual const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) { return NULL; };

    /**
        The function must make sure not to write in d_out[nvalue] and beyond. If decompressed size is greater than
        the capacity (nvalue), there is no need to write any output at all.
    **/
    __device__ virtual const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) = 0;

    /**
     * Decompress compressed list using thread parallelism.
    **/
    __device__ virtual const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) = 0;

    virtual __device__ __host__
    ~DeviceIntegerCODEC() {}

    /** Convenience function not supported */
    virtual std::vector<uint32_t>
    compress(std::vector<uint32_t> &data) {
        throw std::logic_error("DeviceIntegerCODEC::compress not supported!");
    }

    /** Convenience function not supported */
    virtual std::vector<uint32_t>
    uncompress(std::vector<uint32_t> &compresseddata, size_t expected_uncompressed_size = 0) {
        throw std::logic_error("DeviceIntegerCODEC::uncompress not supported!");
    }

    virtual std::string
    name() const = 0;
};

class DeviceJustCopyCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceJustCopyCodec() : DeviceIntegerCODEC() {}

    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        nvalue = length;
    }

    virtual const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        nvalue = length;
        return in + length;
    }

    __device__ virtual const uint32_t*
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

    __device__ virtual const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
    {
        assert(length <= gridDim.x * blockDim.x); // 1 thread copies one value
        assert(length <= nvalue); // not enough capacity in the decompressed array!

        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        d_out[idx] = d_in[idx];

        nvalue = length;
        return d_in + length;
    }

    virtual __device__ __host__
    ~DeviceJustCopyCodec() {}

    virtual std::string
    name() const { return "DeviceJustCopyCodec"; }
};


class DeviceDeltaCodec : public DeviceIntegerCODEC {
public:

    __device__ __host__
    DeviceDeltaCodec() : DeviceIntegerCODEC() {}

    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        DeviceDeltaHelper<uint32_t>::delta(0, out, length);
        nvalue = length;
    }

    virtual const uint32_t*
    decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        std::memcpy(out, in, sizeof(uint32_t) * length);
        DeviceDeltaHelper<uint32_t>::inverseDelta(0, out, length);
        nvalue = length;
        return in + length;
    }

    __device__ virtual const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
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

    __device__ virtual const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue)
    {
        assert(length <= gridDim.x * blockDim.x * 4); // one thread can process 4 values
        assert(length <= nvalue); // not enough capacity in the decompressed array!
        assert(blockIdx.x == 0) // currently only support single block

        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        d_out[idx] = 0; // d_out should be shared memory

        size_t arrayLength = (length + 3) / 4;
        assert((arrayLength >= GPUGENIE_SCAN_MIN_SHORT_ARRAY_SIZE) &&
                (arrayLength <= GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE));
        uint pow2arrayLength = pow2ceil_32(arrayLength);

        // Check supported size range
        // Check parallel model compatibility
        assert(blockDim.x == GPUGENIE_SCAN_THREADBLOCK_SIZE && gridDim.x == 1);

        __syncthreads();
        d_scanExclusiveShared((uint4 *)d_in, (uint4 *)d_out, arrayLength / 4, pow2arrayLength);
        __syncthreads();
        
        if (idx == 0)
            assert(d_out[idx] == 0);
        else
            assert(d_out[idx] >= d_out[idx]-1);

        // turn it into inclusive scan
        uint32_t inc = d_in[0];
        __syncthreads();
        d_out[idx] += inc;
        __syncthreads();

        nvalue = length;
        return d_in + length;
    }

    virtual __device__ __host__
    ~DeviceDeltaCodec() {}

    virtual std::string
    name() const { return "DeviceDeltaCodec"; }
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

    virtual __device__ __host__
    ~DeviceCompositeCodec() {}

    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
        const size_t roundedlength = length / Codec1::BlockSize * Codec1::BlockSize;
        size_t nvalue1 = nvalue;
        codec1.encodeArray(in, roundedlength, out, nvalue1);

        if (roundedlength < length) {
            ASSERT(nvalue >= nvalue1, nvalue << " " << nvalue1);
            size_t nvalue2 = nvalue - nvalue1;
            codec2.encodeArray(in + roundedlength, length - roundedlength, out + nvalue1, nvalue2);
            nvalue = nvalue1 + nvalue2;
        } else {
            nvalue = nvalue1;
        }
    }

    virtual const uint32_t*
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

    __device__ virtual const uint32_t*
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

    __device__ virtual const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t length, uint32_t *d_out, size_t &nvalue) {
        return NULL;
    }

    std::string name() const {
        std::ostringstream convert;
        convert << "DeviceCompositeCodec_" << codec1.name() << "+" << codec2.name();
        return convert.str();
    }
};

}

#endif
