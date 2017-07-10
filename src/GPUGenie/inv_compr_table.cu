#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include "Logger.h"
#include "genie_errors.h"
#include "Timing.h"
#include "DeviceCodecFactory.h"
#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"

#include "inv_compr_table.h"

BOOST_CLASS_EXPORT_IMPLEMENT(GPUGenie::inv_compr_table)

void
GPUGenie::inv_compr_table::build(size_t max_length, bool use_load_balance)
{
    Logger::log(Logger::DEBUG, "Bulding uncompressed inv_table...");
    inv_table::build(max_length, use_load_balance);


    Logger::log(Logger::DEBUG, "Bulding compressed inv_table...");

    std::vector<int> &inv = *(inv_table::inv());
    std::vector<int> &invPos = *(inv_table::inv_pos());
    std::vector<uint32_t> &compressedInv = m_comprInv;
    std::vector<int> &compressedInvPos = m_comprInvPos;

    // make uint32_t copy of uncompressed inv array
    // codecs are expecting unsigned integer arrays, but inv_table uses int by default
    std::vector<uint32_t> inv_u32(inv.begin(), inv.end());

    uint64_t compressionStartTime = getTime();

    // Retrieve coded instance
    std::shared_ptr<DeviceIntegerCODEC> codec = DeviceCodecFactory::getCodec(m_compression);
    if(!codec.get()) {
        Logger::log(Logger::ALERT, "No matching function for %s compression!",
                DeviceCodecFactory::getCompressionName(m_compression).c_str());
        throw std::logic_error("No compression codec available!");
    }
    // Check if codec will be able to decompress an inverted list of any length
    assert(codec->decodeArrayParallel_lengthPerBlock() >= (int)max_length);

    compressedInv.resize(inv.size()*8);
    compressedInvPos.clear();
    compressedInvPos.reserve(invPos.size());
    compressedInvPos.push_back(0);

    int compressedInvSize = 0;
    int64_t compressedInvCapacity = compressedInv.size();
    int badCompressedLists = 0;

    uint32_t *out = compressedInv.data();
    for (int pos = 0; pos < (int)invPos.size()-1; pos++)
    {
        int invStart = invPos[pos];
        int invEnd = invPos[pos+1];
        assert(invEnd - invStart > 0 && invEnd - invStart <= (int)max_length);

        // Check if we run out of capacity 
        assert(compressedInvCapacity > 0);
        // We cannot have more capacity then there is free space in the output vector compressedInv, plus at the same
        // time we cannot have negative compression overflow our max_length constraint on inverted list 
        size_t nvalue = std::min((size_t)compressedInvCapacity, max_length);

        uint32_t * data = inv_u32.data() + invStart;
        codec->encodeArray(data, invEnd - invStart, out, nvalue);

        // Check if the compressed length (nvalue) from encodeArray(...) does not exceed the max_length constraint
        // of the compressed list
        assert(nvalue > 0 && nvalue <= max_length);

        out += nvalue; // shift compression output pointer
        compressedInvCapacity -= nvalue;
        compressedInvSize += nvalue;

        compressedInvPos.push_back(compressedInvSize);

        if ((int)nvalue >= invEnd - invStart)
            badCompressedLists++;
    }

    for (size_t i = 1; i < compressedInvPos.size(); i++){
        assert(compressedInvPos[i] > compressedInvPos[i-1]); // Check if there was no int overflow in compressedInvPos
    }

    compressedInv.resize(compressedInvSize); // shrink to used space only
    compressedInv.shrink_to_fit();
    assert(compressedInvSize == compressedInvPos.back());

    uint64_t compressionEndTime = getTime();
    double compressionRatio = 32.0 * static_cast<double>(compressedInv.size()) / static_cast<double>(inv.size());

    Logger::log(Logger::DEBUG, "Done bulding compressed inv_compr_table in time %f",
        getInterval(compressionStartTime, compressionEndTime));

    Logger::log(Logger::INFO, "Compression %s, codec: %s, compression ratio: %f",
        DeviceCodecFactory::getCompressionName(m_compression).c_str(), codec->name().c_str(), compressionRatio);

    if (compressionRatio > 16.0 || badCompressedLists)
        Logger::log(Logger::ALERT, "Bad compression! Bad compressed lists: %d / %d, compression ratio: %f",
            badCompressedLists, compressedInvPos.size()-1, compressionRatio);
}


GPUGenie::inv_compr_table::~inv_compr_table()
{
    clear_gpu_mem();
}


GPUGenie::COMPRESSION_TYPE
GPUGenie::inv_compr_table::getCompression() const
{
    return m_compression;
}

double
GPUGenie::inv_compr_table::getCompressionRatio()
{
    if (this->build_status() != builded)
    {
        Logger::log(Logger::ALERT, "Unknown compression ratio: table is not built!");
        return -1;
    }
    assert(m_comprInv.size());
    assert(uncompressedInv()->size());
    return 32.0 * static_cast<double>(m_comprInv.size()) / static_cast<double>(uncompressedInv()->size());
}

void
GPUGenie::inv_compr_table::setCompression(COMPRESSION_TYPE compression)
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
GPUGenie::inv_compr_table::inv()
{
    return reinterpret_cast<std::vector<int>*>(&m_comprInv);
}

std::vector<uint32_t>*
GPUGenie::inv_compr_table::compressedInv()
{
    return &m_comprInv;
}

std::vector<int>*
GPUGenie::inv_compr_table::uncompressedInv()
{
    return inv_table::inv();
}

std::vector<int>*
GPUGenie::inv_compr_table::inv_pos()
{
    return &m_comprInvPos;
}
std::vector<int>*
GPUGenie::inv_compr_table::compressedInvPos()
{
    return &m_comprInvPos;
}

std::vector<int>*
GPUGenie::inv_compr_table::uncompressedInvPos()
{
    return inv_table::inv_pos();;
}

uint32_t*
GPUGenie::inv_compr_table::deviceCompressedInv() const
{
    return m_d_compr_inv_p;
}

bool GPUGenie::inv_compr_table::cpy_data_to_gpu()
{
    try{
        if(m_d_compr_inv_p == NULL)
            cudaCheckErrors(cudaMalloc(&m_d_compr_inv_p, sizeof(uint32_t) * m_comprInv.size()));
        cudaCheckErrors(cudaMemcpy(m_d_compr_inv_p, &m_comprInv[0], sizeof(uint32_t) * m_comprInv.size(),
                cudaMemcpyHostToDevice));
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
