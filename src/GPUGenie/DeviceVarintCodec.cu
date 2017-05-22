#include "DeviceVarintCodec.h"

using namespace GPUGenie;

void
GPUGenie::DeviceVarintCodec::encodeArray(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{

    uint8_t *bout = reinterpret_cast<uint8_t *>(out);
    const uint8_t *const initbout = reinterpret_cast<uint8_t *>(out);
    size_t bytenvalue = nvalue * sizeof(uint32_t);
    encodeToByteArray(in, length, bout, bytenvalue);
    bout += bytenvalue;
    while (needPaddingTo32Bits(bout)) {
        *bout++ = 0;
    }
    const size_t storageinbytes = bout - initbout;
    assert((storageinbytes % 4) == 0);
    nvalue = storageinbytes / 4;

}

void
GPUGenie::DeviceVarintCodec::encodeToByteArray(uint32_t *in, const size_t length, uint8_t *bout, size_t &nvalue) {
    const uint8_t *const initbout = bout;
    for (size_t k = 0; k < length; ++k) {
        const uint32_t val = in[k];

        if (val < (1U << 7)) {
            *bout = static_cast<uint8_t>(val | (1U << 7));
            ++bout;
        } else if (val < (1U << 14)) {
            *bout = extract7bits<0>(val);
            ++bout;
            *bout = extract7bitsmaskless<1>(val) | (1U << 7);
            ++bout;
        } else if (val < (1U << 21)) {
            *bout = extract7bits<0>(val);
            ++bout;
            *bout = extract7bits<1>(val);
            ++bout;
            *bout = extract7bitsmaskless<2>(val) | (1U << 7);
            ++bout;
        } else if (val < (1U << 28)) {
            *bout = extract7bits<0>(val);
            ++bout;
            *bout = extract7bits<1>(val);
            ++bout;
            *bout = extract7bits<2>(val);
            ++bout;
            *bout = extract7bitsmaskless<3>(val) | (1U << 7);
            ++bout;
        } else {
            *bout = extract7bits<0>(val);
            ++bout;
            *bout = extract7bits<1>(val);
            ++bout;
            *bout = extract7bits<2>(val);
            ++bout;
            *bout = extract7bits<3>(val);
            ++bout;
            *bout = extract7bitsmaskless<4>(val) | (1U << 7);
            ++bout;
        }
    }
    nvalue = bout - initbout;
}

const uint32_t*
GPUGenie::DeviceVarintCodec::decodeArray(const uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue)
{
    decodeFromByteArray((const uint8_t *)in, length * sizeof(uint32_t), out, nvalue);
    return in + length;
}



const uint8_t*
GPUGenie::DeviceVarintCodec::decodeFromByteArray(const uint8_t *inbyte, const size_t length, uint32_t *out,
        size_t &nvalue)
{
    if (length == 0) {
        nvalue = 0;
        return inbyte;
    }
    const uint8_t *const endbyte = inbyte + length;
    const uint32_t *const initout(out);

    while (endbyte > inbyte + 5) {

        uint8_t c;
        uint32_t v;

        c = inbyte[0];
        v = c & 0x7F;
        if (c >= 128) {
            inbyte += 1;
            *out++ = v;
            continue;
        }

        c = inbyte[1];
        v |= (c & 0x7F) << 7;
        if (c >= 128) {
            inbyte += 2;
            *out++ = v;
            continue;
        }

        c = inbyte[2];
        v |= (c & 0x7F) << 14;
        if (c >= 128) {
            inbyte += 3;
            *out++ = v;
            continue;
        }

        c = inbyte[3];
        v |= (c & 0x7F) << 21;
        if (c >= 128) {
            inbyte += 4;
            *out++ = v;
            continue;
        }

        c = inbyte[4];
        inbyte += 5;
        v |= (c & 0x0F) << 28;
        *out++ = v;
    }
    while (endbyte > inbyte) {
        unsigned int shift = 0;
        for (uint32_t v = 0; endbyte > inbyte; shift += 7) {
            uint8_t c = *inbyte++;
            v += ((c & 127) << shift);
            if ((c & 128)) {
                *out++ = v;
                break;
            }
        }
    }
    nvalue = out - initout;
    return inbyte;
}


__device__ uint32_t*
GPUGenie::DeviceVarintCodec::decodeArraySequential(
    uint32_t *d_in, size_t comprLength, uint32_t *d_out, size_t &nvalue)
{
    return nullptr;
}

__device__ uint32_t*
GPUGenie::DeviceVarintCodec::decodeArrayParallel(
    uint32_t *d_in, size_t comprLength, uint32_t *d_out, size_t &capacity)
{
    int idx = threadIdx.x;

    assert(comprLength > 0);
    assert(comprLength <=  GPUGENIE_CODEC_VARINT_MAX_UNCOMPR_LENGTH);

    // each thread stores number of integers that are decoded from the uint32_t processed by the thread
    // the possible values in this array may be 1..4
    __shared__ uint32_t s_numInts[GPUGENIE_CODEC_VARINT_MAX_UNCOMPR_LENGTH];
    __shared__ uint32_t s_numIntsScanned[GPUGENIE_CODEC_VARINT_MAX_UNCOMPR_LENGTH];

    for (int i = 0; i < (comprLength + GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE - 1) /
        GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE; i++)
    {
        int idxUnpack = i * GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE + idx;
        
        if (idxUnpack < comprLength)
            s_numInts[idxUnpack] = numIntsStartingHere(d_in, idxUnpack, comprLength);
        else
            s_numInts[idxUnpack] = 0;
    }

    // do a scan of s_numInts to find d_out position for each thread
    uint comprLengthPow2 = GPUGenie::d_pow2ceil_32(comprLength);
    uint comprLength4 = (comprLength + 3) / 4;
    __syncthreads();
    GPUGenie::d_scanExclusivePerBlockShared((uint4 *)s_numIntsScanned, (uint4 *)s_numInts, comprLength4, comprLengthPow2);
    __syncthreads();

    int decomprLength = s_numIntsScanned[comprLength-1] + s_numInts[comprLength-1];
    assert(decomprLength <= capacity);

    // we need at most 4 loops of unpacking for the current setup, since we use exactly 256 threads,
    // but the maximal unpacked capacity is 1024
    for (int i = 0; i < (comprLength + GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE - 1) /
        GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE; i++)
    {
        int idxUnpack = i * GPUGENIE_CODEC_VARINT_THREADBLOCK_SIZE + idx;

        if (idxUnpack >= comprLength)
            break;

        uint8_t* nextByte = reinterpret_cast<uint8_t*>(d_in + idxUnpack);
        uint8_t myCurrByte = *nextByte++;
        uint8_t myPrevByte = idxUnpack > 0 ? (d_in[idxUnpack-1] >> 24) : 0xFF;

        int myNumInts = (int)s_numInts[idxUnpack];
        int myOutIdx = (int)s_numIntsScanned[idxUnpack];

        assert(myNumInts <= 4);
        assert(myOutIdx < decomprLength || myNumInts == 0);

        // find first starting position position, such that previous byte has 1 on the highest position (last byte of
        // it's corresponding int)
        while (!(myPrevByte & 128))
        {
            myPrevByte = myCurrByte;
            myCurrByte = *nextByte++;
        }

        for (int i = 0; i < myNumInts; i++)
        {
            uint32_t decoded = 0;
            for (unsigned int shift = 0; ; shift += 7)
            {
                decoded += (myCurrByte & 127) << shift;
                if (myCurrByte & 128) // this was last byte
                {
                    d_out[myOutIdx + i] = decoded;
                    // printf("Thread: %d unpacked idx: %d int number %d out of numInts %d, value: %u, saved into d_out[%d]\n", idx, idxUnpack, i, myNumInts, decoded, myOutIdx + i);
                    myCurrByte = *nextByte++;
                    break;
                }
                myCurrByte = *nextByte++;
            }
        }
    }

    capacity = decomprLength;
    return d_in + comprLength;

}

__device__ int
GPUGenie::DeviceVarintCodec::numIntsStartingHere(uint32_t *d_in, int idxUnpack, int comprLength)
{
    // This function checks the last byte of the preceding uint32_t and the first 3 bytes of the current uint32_t, i.e.
    // d_in[idxUnpack]. If such byte value has 1 in the highest bit, then a new int must start in this uint32_t
    uint8_t* nextBytePtr = reinterpret_cast<uint8_t*>(d_in + idxUnpack);
    uint8_t prevByte = idxUnpack > 0 ? (d_in[idxUnpack-1] >> 24) : 0xFF;
    int numIntsStartingHere = 0;

    for (int i = 0; i < 4; i++)
    {
        if (prevByte & 128)
            numIntsStartingHere++;

        prevByte = *nextBytePtr;
        nextBytePtr++;
    }
    // the last compressed uint32_t may have leading bits of some bytes set to 0, even though no integer starts there
    if (idxUnpack == comprLength - 1 && !(prevByte & 128))
        numIntsStartingHere--;
    return numIntsStartingHere;
}
