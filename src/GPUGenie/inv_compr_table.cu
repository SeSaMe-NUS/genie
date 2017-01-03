#include <iostream>

#include "Logger.h"
#include "genie_errors.h"
#include "Timing.h"

#include "inv_compr_table.h"


void
GPUGenie::inv_compr_table::build(u64 max_length, bool use_load_balance)
{
    Logger::log(Logger::DEBUG, "Bulding uncompressed inv_table...");
    inv_table::build(max_length, use_load_balance);


    Logger::log(Logger::DEBUG, "Bulding compressed inv_table...");

    std::vector<int> &inv = *(inv_table::inv());
    std::vector<int> &invPos = *(inv_table::inv_pos());
    std::vector<uint32_t> &compressedInv = m_comprInv;
    std::vector<int> &compressedInvPos = m_comprInvPos;

    uint64_t compressionStartTime = getTime();

    shared_ptr<DeviceIntegerCODEC> codec;
    switch (this->m_compression){
        case "copy":
            codec = std::shared_ptr<IntegerCODEC>(
                        new DeviceJustCopy());
            break;
        case "d1":
            codec = std::shared_ptr<IntegerCODEC>(
                        new DeviceDeltaCodec());
            break;
        case "d1-bp32":
            codec = std::shared_ptr<IntegerCODEC>(
                        new DeviceCompositeCodec<DeviceBinaryPacking,DeviceJustCopy>());
            break;
        case "d1-varint":
        case "d1-bp32-varint":
        default:
            Logger::log(Logger::ALERT, "Unsupported inverted table compression %s", this->m_compression);
    }

    // make uint32_t copy of uncompressed inv array
    std::vector<uint32_t> inv32(inv.begin(), inv.end());

    compressedInv.resize(inv.size());
    compressedInvPos.resize(invPos.size());
    compressedInvPos.push_back(0);

    
    int compressedInvCurrentPos = 0;
    int compressedInvSize = 0;
    int compressedInvCapacity = compressedInv.size();

    uint32_t *out = compressedInv.data();
    for (int pos = 0; pos < invPos.size(); pos++)
    {
        int invStart = invPos[pos];
        int invEnd = invPos[pos+1];
        size_t invLength = invEnd - invStart;
        assert(invEnd > invStart);

        uint32_t *in = inv32.data() + sizeof(uint32_t) * invStart; // compression input
        size_t nvalue = compressedInvCapacity; // nvalue is the compressed size

        codec->encodeArray(in, invLength, out, nvalue);

        out += sizeof(int) * nvalue; // shift compression output pointer
        compressedInvCapacity -= nvalue;
        compressedInvSize += nvalue;

        compressedInvPos.push_back(compressedInvSize);
    }

    compressedInv.resize(compressedInvSize); // shrink to used space only
    assert(compressedInvSize == compressedInvPos.back());

    uint64_t compressionEndTime = getTime();

    Logger::log(Logger::DEBUG, "Done bulding compressed inv_compr_table in time %f",
        getInterval(compressionStartTime, compressionEndTime));

    Logger::log(Logger::INFO, "Compression %s, codec: %s, compression ratio: %f", m_compression.c_str(),
        codec->name(), 32.0 * static_cast<double>(compressedInv.size()) / static_cast<double>(inv.size()));
}


void GPUGenie::inv_compr_table::bp32(uint32_t *in, const size_t length, uint32_t *out, size_t &nvalue) {
    checkifdivisibleby(length, BlockSize);
    const uint32_t *const initout(out);
    *out++ = static_cast<uint32_t>(length);
    uint32_t Bs[HowManyMiniBlocks];
    uint32_t init = 0;
    const uint32_t *const final = in + length;
    for (; in + HowManyMiniBlocks * MiniBlockSize <= final;
         in += HowManyMiniBlocks * MiniBlockSize) {
      uint32_t tmpinit = init;
      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
        Bs[i] = BlockPacker::maxbits(in + i * MiniBlockSize, tmpinit);
      }
      *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
        BlockPacker::packblockwithoutmask(in + i * MiniBlockSize, out, Bs[i],
                                          init);
        out += Bs[i];
      }
    }
    if (in < final) {
      size_t howmany = (final - in) / MiniBlockSize;
      uint32_t tmpinit = init;
      memset(&Bs[0], 0, HowManyMiniBlocks * sizeof(uint32_t));
      for (uint32_t i = 0; i < howmany; ++i) {
        Bs[i] = BlockPacker::maxbits(in + i * MiniBlockSize, tmpinit);
      }
      *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
      for (uint32_t i = 0; i < howmany; ++i) {
        BlockPacker::packblockwithoutmask(in + i * MiniBlockSize, out, Bs[i],
                                          init);
        out += Bs[i];
      }
    }
    nvalue = out - initout;
  }

GPUGenie::inv_compr_table::~inv_compr_table()
{
    clear_gpu_mem();
}


const std::string&
GPUGenie::inv_compr_table::getCompression() const
{
    return m_compression;
}

void
GPUGenie::inv_compr_table::setCompression(const std::string &compression)
{
    if (this->build_status() == builded)
    {
        Logger::log(Logger::ALERT, "ERROR: Attempting to change compression type on already built table!");
        return;
    }
    m_compression = compression;
}

size_t
GPUGenie::inv_compr_table::getUncompressedPostingListMaxLength() const
{
    return m_uncompressedInvListsMaxLength;
}

void
GPUGenie::inv_compr_table::setUncompressedPostingListMaxLength(size_t length)
{
    this->m_uncompressedInvListsMaxLength = length;
}

std::vector<int>*
GPUGenie::inv_compr_table::compressedInv()
{
    return &m_comprInv;
}

std::vector<int>*
GPUGenie::inv_compr_table::compressedInvPos()
{
    return &m_comprInvPos;
}

std::vector<int>*
GPUGenie::inv_compr_table::compressedInvIndex()
{
    return this->inv_index();
}

std::vector<int>*
GPUGenie::inv_compr_table::compressedCK()
{
    return this->ck();
}

int*
GPUGenie::inv_compr_table::deviceCompressedInv() const
{
    return m_d_compr_inv_p;
}

bool GPUGenie::inv_compr_table::cpy_data_to_gpu()
{
    try{
        if(m_d_compr_inv_p == NULL)
            cudaCheckErrors(cudaMalloc(&m_d_compr_inv_p, sizeof(int) * m_comprInv.size()));
        u64 t = getTime();
        cudaCheckErrors(cudaMemcpy(m_d_compr_inv_p, &m_comprInv[0], sizeof(int) * m_comprInv.size(),
                cudaMemcpyHostToDevice));
        u64 tt = getTime();
        std::cout<<"The compressed inverted list(all data) transfer time = " << getInterval(t,tt) << "ms" <<std::endl;

    } catch(std::bad_alloc &e){
        throw(GPUGenie::gpu_bad_alloc(e.what()));
    }

    return true;
}

void GPUGenie::inv_compr_table::clear()
{
    inv_table::clear();

    ck()->clear();
    m_comprInv.clear();
    m_comprInvPos.clear();
}

void GPUGenie::inv_compr_table::clear_gpu_mem()
{
    if (m_d_compr_inv_p == NULL)
        return;

    std::cout << "cudaFreeTime: " ;
    u64 t1 = getTime();
    cudaCheckErrors(cudaFree(m_d_compr_inv_p));
    u64 t2 = getTime();
    std::cout << getInterval(t1, t2) << " ms."<< std::endl;

}

