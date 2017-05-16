#include "DeviceBitPackingCodec.h"

using namespace GPUGenie;

void
GPUGenie::DeviceBitPackingCodec::encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
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
        assert(out - initout <= (int)nvalue);
        nvalue = out - initout;
    }

const uint32_t*
GPUGenie::DeviceBitPackingCodec::decodeArray(const uint32_t *in, const size_t /*len*/, uint32_t *out, size_t &nvalue)
{
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

__device__ uint32_t*
GPUGenie::DeviceBitPackingCodec::decodeArraySequential(
        uint32_t *d_in, size_t /*length*/, uint32_t *d_out, size_t &nvalue)
{
    uint32_t actuallength = *d_in++;
    uint32_t *initout(d_out);
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

__device__ uint32_t*
GPUGenie::DeviceBitPackingCodec::decodeArrayParallel(
        uint32_t *d_in, size_t /* comprLength */, uint32_t *d_out, size_t &capacity)
{
    int idx = threadIdx.x;

    uint32_t length = d_in[0]; // first uint32_t is an uncompressed length
    d_in++;
    assert(length <= decodeArrayParallel_lengthPerBlock()); // one thread can process 4 values
    assert(length <= capacity); // not enough capacity in the decompressed array!
    assert(length > 0);

    uint32_t blocks = (length + GPUGENIE_CODEC_BPP_BLOCK_LENGTH - 1) / GPUGENIE_CODEC_BPP_BLOCK_LENGTH; 

    __shared__ uint32_t s_bitSizes[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];
    __shared__ uint32_t s_bitOffsets[GPUGENIE_CODEC_BPP_MAX_BITSIZES_LENGTH];

    if (idx == 0) // thread 0 has to do all the bit sizes summing sequentially
    {
        uint32_t bitOffsetsAcc = 0;
        int inIt = 0;
        for (int b = 0; b < blocks; b+=4)
        {
            s_bitSizes[b] = static_cast<uint8_t>(d_in[inIt] >> 24);
            s_bitSizes[b+1] = static_cast<uint8_t>(d_in[inIt] >> 16);
            s_bitSizes[b+2] = static_cast<uint8_t>(d_in[inIt] >> 8);
            s_bitSizes[b+3] = static_cast<uint8_t>(d_in[inIt]);

            // account for next block of bitSizes
            bitOffsetsAcc += 1;

            // exclusive scan
            s_bitOffsets[b] = bitOffsetsAcc;
            bitOffsetsAcc += s_bitSizes[b];

            s_bitOffsets[b+1] = bitOffsetsAcc;
            bitOffsetsAcc += s_bitSizes[b+1];

            s_bitOffsets[b+2] = bitOffsetsAcc;
            bitOffsetsAcc += s_bitSizes[b+2];

            s_bitOffsets[b+3] = bitOffsetsAcc;
            bitOffsetsAcc += s_bitSizes[b+3];

            // printf("Block %d has bitSize %u and bitOffset %u \n", b, s_bitSizes[b], s_bitOffsets[b]);
            // printf("Block %d has bitSize %u and bitOffset %u \n", b+1, s_bitSizes[b+1], s_bitOffsets[b+1]);
            // printf("Block %d has bitSize %u and bitOffset %u \n", b+2, s_bitSizes[b+2], s_bitOffsets[b+2]);
            // printf("Block %d has bitSize %u and bitOffset %u \n", b+3, s_bitSizes[b+3], s_bitOffsets[b+3]);

            assert(s_bitSizes[b]   <= 32); // bit size has to be in [0,32] range
            assert(s_bitSizes[b+1] <= 32); // bit size has to be in [0,32] range
            assert(s_bitSizes[b+2] <= 32); // bit size has to be in [0,32] range
            assert(s_bitSizes[b+3] <= 32); // bit size has to be in [0,32] range

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
        int blockNum = idxUnpack / GPUGENIE_CODEC_BPP_BLOCK_LENGTH;

        // read the bit size to unpack
        int bitSize = s_bitSizes[blockNum];

        // determine the index of the first and last (exclusive) bit that belongs to the packed number
        int firstBit = bitSize * (idxUnpack % GPUGENIE_CODEC_BPP_BLOCK_LENGTH);
        int lastBit = firstBit + bitSize - 1;
        assert(lastBit < bitSize * GPUGENIE_CODEC_BPP_BLOCK_LENGTH); // cannot exceed bit packed size

        // choose a packed bit range(s)
        uint32_t packed = d_in[s_bitOffsets[blockNum] + firstBit / GPUGENIE_CODEC_BPP_BLOCK_LENGTH]; 
        int firstBitInPacked = firstBit % 32;
        uint32_t packedOverflow = d_in[s_bitOffsets[blockNum] + lastBit / GPUGENIE_CODEC_BPP_BLOCK_LENGTH];
        // assert(lastBit % 32 != firstBitInPacked);

        bool isOverflowing = lastBit % 32 < firstBitInPacked;
        int lastBitInPacked = isOverflowing ? 31 : lastBit % 32;
        int lastBitInPackedOverflow = !isOverflowing ? -1 : lastBit % 32;

        // compute decompressed value
        uint32_t outFromPacked = 
            ((packed >> firstBitInPacked) & (0xFFFFFFFF >> (32 - (bitSize - lastBitInPackedOverflow - 1))));
        uint32_t outFromOverflow = 
            (packedOverflow & (0xFFFFFFFF >> (32-lastBitInPackedOverflow-1))) << (bitSize-lastBitInPackedOverflow-1); 
        uint32_t out = outFromPacked | outFromOverflow;
                       
        d_out[idxUnpack] = out;

        // printf("Thread %d unpacked idx %d: bitSize: %d, firstBit: %d, lastBit: %d, firstBitInPacked: %d, lastBitInPacked: %d, lastBitInPackedOverflow: %d, bits in packed: %d, bits in overflow: %d, out: %u\n", idx, idxUnpack, bitSize, firstBit, lastBit, firstBitInPacked, lastBitInPacked, lastBitInPackedOverflow, bitSize - lastBitInPackedOverflow - 1, lastBitInPackedOverflow, out);
    }

    capacity = length;
    int lastBlock = blocks - 1;
    int offsetPastLastBlock = s_bitOffsets[lastBlock] + s_bitSizes[lastBlock];
    return d_in + offsetPastLastBlock;
}
