#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include "Logger.h"
#include "genie_errors.h"
#include "Timing.h"
#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"

#include "inv_compr_table.h"


std::map<std::string, std::shared_ptr<GPUGenie::DeviceIntegerCODEC>>
GPUGenie::inv_compr_table::initCodecs() {
  std::map<std::string, shared_ptr<DeviceIntegerCODEC>> codecs;

  codecs["copy"] = std::shared_ptr<DeviceCopyCodec>(
                        new DeviceCopyCodec());
  codecs["d1"] = std::shared_ptr<DeviceDeltaCodec>(
                        new DeviceDeltaCodec());
  codecs["bp32"] = std::shared_ptr<DeviceBitPackingCodec>(
                        new DeviceBitPackingCodec());
  codecs["varint"] = std::shared_ptr<DeviceVarintCodec>(
                        new DeviceVarintCodec());
  codecs["bp32-copy"] = std::shared_ptr<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>(
                        new DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>());
  codecs["bp32-varint"] = std::shared_ptr<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>(
                        new DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>());
  return codecs;
}

std::map<std::string, std::shared_ptr<GPUGenie::DeviceIntegerCODEC>>
GPUGenie::inv_compr_table::m_codecs = GPUGenie::inv_compr_table::initCodecs();

std::shared_ptr<GPUGenie::DeviceIntegerCODEC> GPUGenie::inv_compr_table::getCodec(const std::string &codecId)
{
    if(m_codecs.find(codecId) == m_codecs.end()) {
        Logger::log(Logger::ALERT, "Unknown codec %s.", codecId.c_str());
        return std::shared_ptr<GPUGenie::DeviceIntegerCODEC>(nullptr);
    }
    return m_codecs[codecId];
}



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

    shared_ptr<DeviceIntegerCODEC> codec;
    // Retrieve coded based on compression string 
    if(m_codecs.find(this->m_compression) == m_codecs.end()) {
        Logger::log(Logger::ALERT, "Unsupported inverted table compression %s.", this->m_compression.c_str());
        throw std::logic_error("No compression codec available!");
    }
    codec = m_codecs[this->m_compression];
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

    Logger::log(Logger::INFO, "Compression %s, codec: %s, compression ratio: %f", m_compression.c_str(),
        codec->name().c_str(), compressionRatio);

    if (compressionRatio > 16.0 || badCompressedLists)
        Logger::log(Logger::ALERT, "Bad compression! Bad compressed lists: %d / %d, compression ratio: %f",
            badCompressedLists, compressedInvPos.size()-1, compressionRatio);
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
        u64 t = getTime();
        cudaCheckErrors(cudaMemcpy(m_d_compr_inv_p, &m_comprInv[0], sizeof(uint32_t) * m_comprInv.size(),
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


bool
GPUGenie::inv_compr_table::write_to_file(ofstream& ofs)
{
    inv_table::write_to_file(ofs);

    ofs.write((char*)&m_isCompressed, sizeof(bool));
    ofs.write((char*)m_compression.c_str(), sizeof(char)*m_compression.size()+1);
    ofs.write((char*)&m_uncompressedInvListsMaxLength, sizeof(size_t));

    size_t comprInvSize = m_comprInv.size();
    size_t comprInvPosSize = m_comprInvPos.size();
    ofs.write((char*)&comprInvSize, sizeof(size_t));
    ofs.write((char*)&comprInvPosSize, sizeof(size_t));
    ofs.write((char*)m_comprInv.data(), sizeof(uint32_t)*m_comprInv.size());
    ofs.write((char*)m_comprInvPos.data(), sizeof(int)*m_comprInvPos.size());

    return true;
}


bool
GPUGenie::inv_compr_table::read_from_file(ifstream& ifs)
{
    inv_table::read_from_file(ifs);

    ifs.read((char*)&m_isCompressed, sizeof(bool));
    std::getline(ifs, m_compression, '\0');
    ifs.read((char*)&m_uncompressedInvListsMaxLength, sizeof(size_t));

    size_t comprInvSize;
    size_t comprInvPosSize;
    ifs.read((char*)&comprInvSize, sizeof(size_t));
    ifs.read((char*)&comprInvPosSize, sizeof(size_t));

    m_comprInv.resize(comprInvSize);
    m_comprInvPos.resize(comprInvPosSize);
    ifs.read((char*)m_comprInv.data(), sizeof(uint32_t)*m_comprInv.size());
    ifs.read((char*)m_comprInvPos.data(), sizeof(int)*m_comprInvPos.size());
    
    return true;
}


bool
GPUGenie::inv_compr_table::read(const char* filename, inv_compr_table*& table)
{
    ifstream ifs(filename, ios::binary|ios::in);
    if(!ifs.is_open())
        return false;
    
    int _table_index, _total_num_of_table;
    ifs.read((char*)&_table_index, sizeof(int));
    ifs.read((char*)&_total_num_of_table, sizeof(int));
    ifs.close();
    if(_table_index!=0 || _total_num_of_table<1)
        return false;
    
    table = new inv_compr_table[_total_num_of_table];
    ifstream _ifs(filename, ios::binary|ios::in);
    
    bool success;
    for(int i=0 ; i<_total_num_of_table ; ++i)
    {
         success = table[i].read_from_file(_ifs);
    }
    _ifs.close();
    return !_ifs.is_open() && success;
}

