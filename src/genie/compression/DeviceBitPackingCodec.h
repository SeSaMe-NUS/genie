#ifndef DEVICE_BIT_PACKING_CODEC_H_
#define DEVICE_BIT_PACKING_CODEC_H_

#include "DeviceCodecs.h"
#include "DeviceBitPackingHelpers.h"

#include <genie/utility/scan.h>

namespace genie
{
namespace compression
{

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

    struct DeviceIntegratedBlockPacker {

        static uint32_t
        maxbits(const uint32_t *in, uint32_t &initoffset) {
            uint32_t accumulator = in[0] - initoffset;
            for (uint32_t k = 1; k < BlockSize; ++k) {
                accumulator |= in[k] - in[k - 1];
            }
            initoffset = in[BlockSize - 1];
            return DeviceBitPackingHelpers::gccbits(accumulator);
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
            return DeviceBitPackingHelpers::gccbits(accumulator);
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
    encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue);

    const uint32_t*
    decodeArray(const uint32_t *in, const size_t /*length*/, uint32_t *out, size_t &nvalue);

    __device__ __host__ static bool
    divisibleby(size_t a, uint32_t x) {
        return (a % x == 0);
    }

    __device__ uint32_t*
    decodeArraySequential(uint32_t *d_in, size_t /*length*/, uint32_t *d_out, size_t &nvalue);

    __device__ uint32_t*
    decodeArrayParallel(uint32_t *d_in, size_t /* comprLength */, uint32_t *d_out, size_t &capacity);

    std::string
    name() const { return "BitPacking32"; }

    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_minEffectiveLength() { return 8; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 256; }
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

    static uint32_t
    maxbits(const uint32_t *in, uint32_t &) {
        uint32_t accumulator = 0;
        for (uint32_t k = 0; k < BlockSize; ++k) {
            accumulator |= in[k];
        }
        return DeviceBitPackingHelpers::gccbits(accumulator);
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
            genie::utility::d_scanExclusiveShared( 
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

    std::string
    name() const { return "BitPacking32Prefixed"; }


    __device__ __host__ int decodeArrayParallel_maxBlocks() { return 1; }
    __device__ __host__ int decodeArrayParallel_lengthPerBlock() { return 1024; }
    __device__ __host__ int decodeArrayParallel_threadsPerBlock() { return 256; }
    __device__ __host__ int decodeArrayParallel_threadLoad() { return 4; }
};

} // namespace compression
} // namespace genie

#endif
