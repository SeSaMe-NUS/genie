#ifndef DEVICE_BIT_PACKING_CODEC_H_
#define DEVICE_BIT_PACKING_CODEC_H_

#include <SIMDCAI/include/common.h>
#include <SIMDCAI/include/util.h>
#include <SIMDCAI/include/codecs.h>

#include "DeviceBitPackingHelpers.h"

namespace GPUGenie {


/**
 *  This class defines an integer codec for binary packing of blocks of 32 integers.
 *  Based on BlockPacker. By default uses inegrated delta encoding.
 *
 *  Support decoding from withing a CUDA kernel.
 */
class DeviceBitPackingCodec : public DeviceIntegerCODEC {

public:
    static const uint32_t MiniBlockSize = 32;
    static const uint32_t HowManyMiniBlocks = 4;
    static const uint32_t BlockSize = MiniBlockSize; // HowManyMiniBlocks * MiniBlockSize;
    static const uint32_t bits32 = 8;

    struct DeviceIntegratedBlockPacker {

        static uint32_t
        maxbits(const uint32_t *in, uint32_t &initoffset) {
            uint32_t accumulator = in[0] - initoffset;
            for (uint32_t k = 1; k < BlockSize; ++k) {
                accumulator |= in[k] - in[k - 1];
            }
            initoffset = in[BlockSize - 1];
            return SIMDCompressionLib::gccbits(accumulator);
        }

        static void inline
        packblockwithoutmask(const uint32_t *in, uint32_t *out,
                             const uint32_t bit,
                             uint32_t &initoffset) {
            DeviceBitPackingHelpers::integratedfastpackwithoutmask(initoffset, in, out, bit);
            initoffset = *(in + BlockSize - 1);
        }

        __device__ __host__ static void inline
        unpackblock(const uint32_t *in, uint32_t *out,
                    const uint32_t bit, uint32_t &initoffset) {
            DeviceBitPackingHelpers::integratedfastunpack(initoffset, in, out, bit);
            initoffset = *(out + BlockSize - 1);
        }
    };


    struct DeviceNoDeltaBlockPacker {

        static uint32_t
        maxbits(const uint32_t *in, uint32_t &) {
            uint32_t accumulator = 0;
            for (uint32_t k = 0; k < BlockSize; ++k) {
                accumulator |= in[k];
            }
            return SIMDCompressionLib::gccbits(accumulator);
        }

        static void inline
        packblockwithoutmask(uint32_t *in, uint32_t *out, const uint32_t bit, uint32_t &) {
            DeviceBitPackingHelpers::fastpackwithoutmask(in, out, bit);
        }

        __device__ __host__ static void inline
        unpackblock(const uint32_t *in, uint32_t *out, const uint32_t bit, uint32_t &) {
            DeviceBitPackingHelpers::fastunpack(in, out, bit);
        }
    };


    virtual void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
        const uint32_t *const initout(out);
        *out++ = static_cast<uint32_t>(length);
        uint32_t Bs[HowManyMiniBlocks];
        uint32_t init = 0;
        const uint32_t *const final = in + length;
        for (; in + HowManyMiniBlocks * MiniBlockSize <= final; in += HowManyMiniBlocks * MiniBlockSize) {
            uint32_t tmpinit = init;
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                Bs[i] = DeviceNoDeltaBlockPacker::maxbits(in + i * MiniBlockSize, tmpinit);
            }
            *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                DeviceNoDeltaBlockPacker::packblockwithoutmask(in + i * MiniBlockSize, out, Bs[i], init);
                out += Bs[i];
            }
        }
        if (in < final) {
            size_t howmany = ((final - in) + MiniBlockSize -1) / MiniBlockSize;
            uint32_t zeroedIn[HowManyMiniBlocks * MiniBlockSize];
            if (!divisibleby(length, BlockSize)) {
                // We treat the rest of the block as 0
                assert(final < in + HowManyMiniBlocks * MiniBlockSize);
                memset(&zeroedIn[0], 0, HowManyMiniBlocks * MiniBlockSize * sizeof(uint32_t));
                memcpy(&zeroedIn[0], in, (final - in) * sizeof(uint32_t));
                assert(zeroedIn[HowManyMiniBlocks * MiniBlockSize - 1] == 0);
                assert(zeroedIn[(final-in)] == 0);
                assert(zeroedIn[(final-in)-1] == in[length-1]);
                in = zeroedIn;
            }
            uint32_t tmpinit = init;
            memset(&Bs[0], 0, HowManyMiniBlocks * sizeof(uint32_t));
            for (uint32_t i = 0; i < howmany; ++i) {
                Bs[i] = DeviceNoDeltaBlockPacker::maxbits(in + i * MiniBlockSize, tmpinit);
            }
            *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
            for (uint32_t i = 0; i < howmany; ++i) {
                DeviceNoDeltaBlockPacker::packblockwithoutmask(in + i * MiniBlockSize, out, Bs[i], init);
                out += Bs[i];
            }
        }
        nvalue = out - initout;
    }

    virtual const uint32_t*
    decodeArray(const uint32_t *in, const size_t /*length*/, uint32_t *out, size_t &nvalue) {
        const uint32_t actuallength = *in++;
        const uint32_t *const initout(out);
        uint32_t Bs[HowManyMiniBlocks];
        uint32_t init = 0;
        for (; out < initout + actuallength / (HowManyMiniBlocks * MiniBlockSize) * HowManyMiniBlocks * MiniBlockSize;
                out += HowManyMiniBlocks * MiniBlockSize) {
            Bs[0] = static_cast<uint8_t>(in[0] >> 24);
            Bs[1] = static_cast<uint8_t>(in[0] >> 16);
            Bs[2] = static_cast<uint8_t>(in[0] >> 8);
            Bs[3] = static_cast<uint8_t>(in[0]);
            ++in;
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                DeviceIntegratedBlockPacker::unpackblock(in, out + i * MiniBlockSize, Bs[i], init);
                in += Bs[i];
            }
        }
        if (out < initout + actuallength) {
            size_t howmany = ((initout + actuallength) - out + MiniBlockSize - 1) / MiniBlockSize;
            Bs[0] = static_cast<uint8_t>(in[0] >> 24);
            Bs[1] = static_cast<uint8_t>(in[0] >> 16);
            Bs[2] = static_cast<uint8_t>(in[0] >> 8);
            Bs[3] = static_cast<uint8_t>(in[0]);
            ++in;

            for (uint32_t i = 0; i < howmany; ++i) {
                DeviceIntegratedBlockPacker::unpackblock(in, out + i * MiniBlockSize, Bs[i], init);
                in += Bs[i];
            }
            if (divisibleby(actuallength, BlockSize))
                out += howmany * MiniBlockSize;
            else
                out += ((initout + actuallength) - out);
        }
        nvalue = out - initout;
        assert(nvalue == actuallength);
        return in;
    }

    __device__ __host__ static bool
    divisibleby(size_t a, uint32_t x) {
        return (a % x == 0);
    }

    __device__ virtual const uint32_t*
    decodeArrayOnGPU(const uint32_t *d_in, const size_t /*length*/, uint32_t *d_out, size_t &nvalue) {
        const uint32_t actuallength = *d_in++;
        const uint32_t *const initout(d_out);
        uint32_t Bs[HowManyMiniBlocks];
        uint32_t init = 0;
        for (;d_out < initout + actuallength / (HowManyMiniBlocks * MiniBlockSize) * HowManyMiniBlocks * MiniBlockSize;
                d_out += HowManyMiniBlocks * MiniBlockSize) {
            Bs[0] = static_cast<uint8_t>(d_in[0] >> 24);
            Bs[1] = static_cast<uint8_t>(d_in[0] >> 16);
            Bs[2] = static_cast<uint8_t>(d_in[0] >> 8);
            Bs[3] = static_cast<uint8_t>(d_in[0]);
            ++d_in;
            for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
                DeviceIntegratedBlockPacker::unpackblock(d_in, d_out + i * MiniBlockSize, Bs[i], init);
                d_in += Bs[i];
            }
        }
        if (d_out < initout + actuallength) {
            size_t howmany = ((initout + actuallength) - d_out + MiniBlockSize - 1) / MiniBlockSize;
            Bs[0] = static_cast<uint8_t>(d_in[0] >> 24);
            Bs[1] = static_cast<uint8_t>(d_in[0] >> 16);
            Bs[2] = static_cast<uint8_t>(d_in[0] >> 8);
            Bs[3] = static_cast<uint8_t>(d_in[0]);
            ++d_in;

            for (uint32_t i = 0; i < howmany; ++i) {
                DeviceIntegratedBlockPacker::unpackblock(d_in, d_out + i * MiniBlockSize, Bs[i], init);
                d_in += Bs[i];
            }
            if (divisibleby(actuallength, BlockSize))
                d_out += howmany * MiniBlockSize;
            else
                d_out += ((initout + actuallength) - d_out);
        }
        nvalue = d_out - initout;
        assert(nvalue == actuallength);
        return d_in;
    }

    virtual __device__ __host__
    ~DeviceBitPackingCodec() {}

    virtual std::string
    name() const { return "DeviceBitPackingCodec"; }
};

}

#endif
