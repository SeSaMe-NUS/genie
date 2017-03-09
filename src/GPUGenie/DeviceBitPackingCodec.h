#ifndef DEVICE_BIT_PACKING_CODEC_H_
#define DEVICE_BIT_PACKING_CODEC_H_

#include <SIMDCAI/include/common.h>
#include <SIMDCAI/include/util.h>
#include <SIMDCAI/include/codecs.h>

#include "DeviceBitPackingHelpers.h"

#include "scan.h"

namespace GPUGenie {

// threadblock size is 256, same for all codecs (can be up to 1024 for compute capability >= 2.0)
#define GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE (256)

// maximum uncompressed length -- read from the first uint32_t of compressed word
#define GPUGENIE_CODEC_BPP_MAX_UNCOMPRESSED_LENGTH (4 * GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE)

// number of integers encoded in a single BP block, each block uses the same bit size
#define GPUGENIE_CODEC_BPP_BLOCK_LENGTH (32)

// maximum number of uint8_t values -- read from the next <length> uint8_ts
#define GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH (GPUGENIE_CODEC_BPP_MAX_UNCOMPRESSED_LENGTH / GPUGENIE_CODEC_BPP_BLOCK_LENGTH)


/**
 *  This class defines an integer codec for binary packing of blocks of 32 integers. The block bit sizes are stored as
 *  4 values uint8_t at the beginning of each block.
 *
 *  Based on BlockPacker from Daniel Lemire.
 *  
 *  By default does not use inegrated delta encoding.
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


    void
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

    const uint32_t*
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
                DeviceNoDeltaBlockPacker::unpackblock(in, out + i * MiniBlockSize, Bs[i], init);
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
                DeviceNoDeltaBlockPacker::unpackblock(in, out + i * MiniBlockSize, Bs[i], init);
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

    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t /*length*/, uint32_t *d_out, size_t &nvalue) {
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
                DeviceNoDeltaBlockPacker::unpackblock(d_in, d_out + i * MiniBlockSize, Bs[i], init);
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
                DeviceNoDeltaBlockPacker::unpackblock(d_in, d_out + i * MiniBlockSize, Bs[i], init);
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

    __device__ const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t /* comprLength */, uint32_t *d_out, size_t &capacity) {
        assert(gridDim.x == 1); // currently only support single block

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        uint32_t length = d_in[0]; // first uint32_t is an uncompressed length
        d_in++;
        assert(length <= gridDim.x * blockDim.x * 4); // one thread can process 4 values
        assert(length <= capacity); // not enough capacity in the decompressed array!
        assert(length > 0);

        uint32_t blocks = (length + GPUGENIE_CODEC_BPP_BLOCK_LENGTH - 1) / GPUGENIE_CODEC_BPP_BLOCK_LENGTH; 

        __shared__ uint32_t s_bitSizes[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];
        __shared__ uint32_t s_bitSizesSummed[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];

        if (idx == 0) // thread 0 has to do all the bit sizes summing sequentially
        {
            uint32_t bitSizesAcc = 0;
            int inIt = 0;
            for (int b = 0; b < blocks; b+=4)
            {
                s_bitSizes[b] = static_cast<uint8_t>(d_in[inIt] >> 24);
                s_bitSizes[b+1] = static_cast<uint8_t>(d_in[inIt] >> 16);
                s_bitSizes[b+2] = static_cast<uint8_t>(d_in[inIt] >> 8);
                s_bitSizes[b+3] = static_cast<uint8_t>(d_in[inIt]);

                // exclusive scan
                s_bitSizesSummed[b] = bitSizesAcc;
                bitSizesAcc += s_bitSizes[b];
                s_bitSizesSummed[b+1] = bitSizesAcc;
                bitSizesAcc += s_bitSizes[b+1];
                s_bitSizesSummed[b+2] = bitSizesAcc;
                bitSizesAcc += s_bitSizes[b+2];
                s_bitSizesSummed[b+3] = bitSizesAcc;
                bitSizesAcc += s_bitSizes[b+3];

                printf("Block %d has bitSize %u\n", b, s_bitSizes[b]);
                printf("Block %d has bitSize %u\n", b+1, s_bitSizes[b+1]);
                printf("Block %d has bitSize %u\n", b+2, s_bitSizes[b+2]);
                printf("Block %d has bitSize %u\n", b+3, s_bitSizes[b+3]);

                assert(s_bitSizes[b]   > 0 && s_bitSizes[b]   <= 32); // bit size has to be in [0,32] range
                assert(s_bitSizes[b+1] > 0 && s_bitSizes[b+1] <= 32); // bit size has to be in [0,32] range
                assert(s_bitSizes[b+2] > 0 && s_bitSizes[b+2] <= 32); // bit size has to be in [0,32] range
                assert(s_bitSizes[b+3] > 0 && s_bitSizes[b+3] <= 32); // bit size has to be in [0,32] range

                // advance the input iterator to another uint32_t with block sizes
                inIt += 1 + s_bitSizes[b] + s_bitSizes[b+1] + s_bitSizes[b+2] + s_bitSizes[b+3];
            }
        }
        __syncthreads();


        // we need at most 4 loops of unpacking for the current setup, since we use exactly 256 threads,
        // but the maximal unpacked capacity is 1024
        for (int i=0; i < (length+GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE-1) / GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE; i++)
        {
            int idxUnpack = i * GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE + idx;

            if (idxUnpack >= length)
                break;

            // every 32 threads process one block
            const uint32_t *d_myIn = d_in + s_bitSizesSummed[idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH];

            // add another input offset due to one uin32_t of block sizes (4 uint8_t values)
            d_myIn += 1 + (idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH);

            // read the bit size to unpack
            int bitSize = s_bitSizes[idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH];

            // determine the index of the first and last (exclusive) bit that belongs to the packed number
            int firstBit = bitSize * (idxUnpack % GPUGENIE_CODEC_BPP_BLOCK_LENGTH);
            int lastBit = firstBit + bitSize;
            assert(lastBit <= bitSize * GPUGENIE_CODEC_BPP_BLOCK_LENGTH); // cannot exceed bit packed size

            // 
            uint32_t packed = d_myIn[firstBit / 32]; // choose a packed source
            int firstBitInPacked = firstBit % 32;
            uint32_t packedOverflow = d_myIn[lastBit / GPUGENIE_CODEC_BPP_BLOCK_LENGTH]; // choose a packed source
            int lastBitInPacked = min(32, lastBit);
            int lastBitInPackedOverflow = max(0, lastBit - 32) % 32;

            uint32_t out = ((packed >> firstBitInPacked) % (1U << bitSize)) |
                           (packedOverflow % (1U << lastBitInPackedOverflow)) << (32 - lastBitInPacked); 

            d_out[idxUnpack] = out;
        }

        capacity = length;
        return d_in + length;
    }

    __device__ __host__
    ~DeviceBitPackingCodec() {}

    std::string
    name() const { return "DeviceBitPackingCodec"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }
};


/**
 *  This class defines an integer codec for binary packing of blocks of 32 integers.
 *  The uncompressed length is stored as first uint32_t value.
 *  The block bit sizes are stored as subsequent values of type uint8_t for all the blocks.
 *
 *  Based on BlockPacker from Daniel Lemire.
 *  
 *  No integrated delta.
 *
 *  Support decoding from withing a CUDA kernel.
 */
class DeviceBitPackingPrefixedCodec : public DeviceIntegerCODEC {

public:
    static const uint32_t MiniBlockSize = 32;
    static const uint32_t HowManyMiniBlocks = 4;
    static const uint32_t BlockSize = MiniBlockSize; // HowManyMiniBlocks * MiniBlockSize;
    static const uint32_t bits32 = 8;

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

    __device__ __host__ static bool inline
    divisibleby(size_t a, uint32_t x) {
        return (a % x == 0);
    }

    void
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
    {
        return;
    }

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t /*length*/, uint32_t *out, size_t &nvalue)
    {
        return NULL;
    }

    __device__ const uint32_t*
    decodeArraySequential(const uint32_t *d_in, const size_t /*length*/, uint32_t *d_out, size_t &nvalue)
    {
        return NULL;
    }

    __device__ const uint32_t*
    decodeArrayParallel(const uint32_t *d_in, const size_t comprLength, uint32_t *d_out, size_t &capacity) {
        assert(gridDim.x == 1); // currently only support single block

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        uint32_t length = d_in[0]; // first uint32_t is an uncompressed length
        d_in++;
        assert(length <= gridDim.x * blockDim.x * 4); // one thread can process 4 values
        assert(length <= capacity); // not enough capacity in the decompressed array!
        assert(length > 0);

        uint32_t blocks = (length + GPUGENIE_CODEC_BPP_BLOCK_LENGTH - 1) / GPUGENIE_CODEC_BPP_BLOCK_LENGTH; 

        __shared__ uint32_t s_bitSizes[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];
        __shared__ uint32_t s_bitSizesSummed[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];

        if (idx < blocks)
        {
            s_bitSizes[idx] = (d_in[idx/4] >> (24 - 8 * (idx % 4))) & 0xFFu;
            printf("Block %d has bitSize %u\n", idx, s_bitSizes[idx]);
            assert(s_bitSizes[idx] > 0 && s_bitSizes[idx] <= 32); // bit size has to be in [0,32] range
        }
        __syncthreads();

        if (blocks > 1)
        {
            // TODO, if length is short, there is no need to do full scan of lenth
            // GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH, instead only scan #block bit sizes 
            d_scanExclusiveShared( 
                    (uint4 *)s_bitSizes,
                    (uint4 *)s_bitSizesSummed,
                    GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH / 4,
                    GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH);
        }
        __syncthreads();

        // we need at most 4 loops of unpacking for the current setup, since we use exactly 256 threads,
        // but the maximal unpacked capacity is 1024
        for (int i=0; i < (length+GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE-1) / GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE; i++)
        {
            int idxUnpack = i * GPUGENIE_CODEC_BPP_THREADBLOCK_SIZE + idx;

            if (idxUnpack >= length)
                break;

            // every 32 threads process one block
            const uint32_t *d_myIn = d_in + s_bitSizesSummed[idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH];

            // read the bit size to unpack
            int bitSize = s_bitSizes[idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH];

            // determine the index of the first and last (exclusive) bit that belongs to the packed number
            int firstBit = bitSize * (idxUnpack % GPUGENIE_CODEC_BPP_BLOCK_LENGTH);
            int lastBit = firstBit + bitSize;
            assert(lastBit <= bitSize * GPUGENIE_CODEC_BPP_BLOCK_LENGTH); // cannot exceed bit packed size

            // 
            uint32_t packed = d_myIn[firstBit / 32]; // choose a packed source
            int firstBitInPacked = firstBit % 32;
            uint32_t packedOverflow = d_myIn[lastBit / GPUGENIE_CODEC_BPP_BLOCK_LENGTH]; // choose a packed source
            int lastBitInPacked = min(32, lastBit);
            int lastBitInPackedOverflow = max(0, lastBit - 32) % 32;

            uint32_t out = ((packed >> firstBitInPacked) % (1U << bitSize)) |
                           (packedOverflow % (1U << lastBitInPackedOverflow)) << (32 - lastBitInPacked); 

            d_out[idxUnpack] = out;
        }

        capacity = length;
        return d_in + length;
    }

    __device__ __host__
    ~DeviceBitPackingPrefixedCodec() {}

    std::string
    name() const { return "DeviceBitPackingPrefixedCodec"; }


    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }
};

}

#endif
