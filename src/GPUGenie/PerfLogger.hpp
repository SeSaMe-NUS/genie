
#ifndef PERF_LOGGER_H_
#define PERF_LOGGER_H_

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "DeviceCodecFactory.h"
#include "Logger.h"

namespace genie
{

namespace util
{

template <typename T>
class PerfLogger{
public:
    static PerfLogger& Instance()
    {
        static PerfLogger<T> pl;
        return pl;
    }

    T& New(const std::string &filename)
    {
        if (ofs_.is_open())
            ofs_.close();
        ofs_.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
        assert(ofs_.good());
        perf_data_->WriteHeader(ofs_);
        return *(perf_data_.get());
    }

    T& Log()
    {
        assert(ofs_.good());
        dirty_perf_data_ = true;
        return *(perf_data_.get());
    }

    T& Reset()
    {
        perf_data_.reset(new T);
        dirty_perf_data_ = false;
        return *(perf_data_.get());
    }

    T& Next()
    {
        assert(ofs_.good());
        // assert(dirty_perf_data_);
        perf_data_->WriteLine(ofs_);
        Reset();
        return *(perf_data_.get());
    }

    virtual ~PerfLogger()
    {
        if (dirty_perf_data_)
            perf_data_->WriteLine(ofs_);
        ofs_.close();
    }

protected:
    PerfLogger() : perf_data_(new T), dirty_perf_data_(false) {};
    PerfLogger(PerfLogger const&);
    PerfLogger(PerfLogger&&);
    PerfLogger<T>& operator=(PerfLogger const&);
    PerfLogger<T>& operator=(PerfLogger&&);

    std::ofstream ofs_;
    std::unique_ptr<T> perf_data_;
    bool dirty_perf_data_;
};


/**
 * PerfLogger requires the following interface from all data classes:
 *
 * class PerfData
 * {
 * public:
 *     PerfData();
 *     void WriteHeader(std::ofstream &ofs) = 0;
 *     void WriteLine(std::ofstream &ofs) = 0;
 *     ~PerfData();
 * };
 */
class MatchingPerfData
{
public:
    MatchingPerfData() :
        compr_(GPUGenie::NO_COMPRESSION),
        overall_time_(0.0),
        query_compilation_time_(0.0),
        preprocessing_time_(0.0),
        query_transfer_time_(0.0),
        data_transfer_time_(0.0),
        constant_transfer_time_(0.0),
        allocation_time_(0.0),
        filling_time_(0.0),
        matching_time_(0.0),
        convert_time_(0.0),
        inv_size_(0),
        dims_size_(0),
        hash_table_capacity_per_query_(0),
        threshold_size_(0),
        passcount_size_(0),
        bitmap_size_(0),
        num_items_in_hash_table_size_(0),
        topks_size_(0),
        hash_table_size_(0),
        compr_ratio_(0.0) {}

    void WriteHeader(std::ofstream &ofs)
    {
        ofs << "codec" << ","
            << "overallTime" << ","
            << "queryCompilationTime" << ","
            << "preprocessingTime" << ","
            << "queryTransferTime" << ","
            << "dataTransferTime" << ","
            << "constantTransferTime" << ","
            << "allocationTime" << ","
            << "fillingTime" << ","
            << "matchingTime" << ","
            << "convertTime" << ","
            << "invSize" << ","
            << "dimsSize" << ","
            << "hashTableCapacityPerQuery" << ","
            << "thresholdSize" << ","
            << "passCountSize" << ","
            << "bitMapSize" << ","
            << "numItemsInHashTableSize" << ","
            << "topksSize" << ","
            << "hashTableSize" << ","
            << "comprRatio" << std::endl;
    }

    void WriteLine(std::ofstream &ofs)
    {
        std::string codec_name("no");
        #ifdef GENIE_COMPR 
            codec_name = GPUGenie::DeviceCodecFactory::getCompressionName(compr_);
        #endif
        ofs << codec_name << ","
            << std::fixed << std::setprecision(3) << overall_time_ << ","
            << std::fixed << std::setprecision(3) << query_compilation_time_ << ","
            << std::fixed << std::setprecision(3) << preprocessing_time_ << ","
            << std::fixed << std::setprecision(3) << query_transfer_time_ << ","
            << std::fixed << std::setprecision(3) << data_transfer_time_ << ","
            << std::fixed << std::setprecision(3) << constant_transfer_time_ << ","
            << std::fixed << std::setprecision(3) << allocation_time_ << ","
            << std::fixed << std::setprecision(3) << filling_time_ << ","
            << std::fixed << std::setprecision(3) << matching_time_ << ","
            << std::fixed << std::setprecision(3) << convert_time_ << ","
            << inv_size_ << ","
            << dims_size_ << ","
            << hash_table_capacity_per_query_ << ","
            << threshold_size_ << ","
            << passcount_size_ << ","
            << bitmap_size_ << ","
            << num_items_in_hash_table_size_ << ","
            << topks_size_ << ","
            << hash_table_size_ << ","
            << std::fixed << std::setprecision(3) << compr_ratio_ << std::endl;
    }

    MatchingPerfData& Compr (GPUGenie::COMPRESSION_TYPE compr)
    {
        compr_ = compr;
        return *this;
    }

    MatchingPerfData& OverallTime (float overall_time)
    {
        overall_time_ = overall_time;
        return *this;
    }

    MatchingPerfData& QueryCompilationTime (float query_compilation_time)
    {
        query_compilation_time_ = query_compilation_time;
        return *this;
    }

    MatchingPerfData& PreprocessingTime (float preprocessing_time)
    {
        preprocessing_time_ = preprocessing_time;
        return *this;
    }

    MatchingPerfData& QueryTransferTime (float query_transfer_time)
    {
        query_transfer_time_ = query_transfer_time;
        return *this;
    }

    MatchingPerfData& DataTransferTime (float data_transfer_time)
    {
        data_transfer_time_ = data_transfer_time;
        return *this;
    }

    MatchingPerfData& ConstantTransferTime (float constant_transfer_time)
    {
        constant_transfer_time_ = constant_transfer_time;
        return *this;
    }

    MatchingPerfData& AllocationTime (float allocation_time)
    {
        allocation_time_ = allocation_time;
        return *this;
    }

    MatchingPerfData& FillingTime (float filling_time)
    {
        filling_time_ = filling_time;
        return *this;
    }

    MatchingPerfData& MatchingTime (float matching_time)
    {
        matching_time_ = matching_time;
        return *this;
    }

    MatchingPerfData& ConvertTime (float convert_time)
    {
        convert_time_ = convert_time;
        return *this;
    }

    MatchingPerfData& InvSize (size_t inv_size)
    {
        inv_size_ = inv_size;
        return *this;
    }

    MatchingPerfData& DimsSize (size_t dims_size)
    {
        dims_size_ = dims_size;
        return *this;
    }

    MatchingPerfData& HashTableCapacityPerQuery (size_t hash_table_capacity_per_query)
    {
        hash_table_capacity_per_query_ = hash_table_capacity_per_query;
        return *this;
    }

    MatchingPerfData& ThresholdSize (size_t threshold_size)
    {
        threshold_size_ = threshold_size;
        return *this;
    }

    MatchingPerfData& PasscountSize (size_t passcount_size)
    {
        passcount_size_ = passcount_size;
        return *this;
    }

    MatchingPerfData& BitmapSize (size_t bitmap_size)
    {
        bitmap_size_ = bitmap_size;
        return *this;
    }

    MatchingPerfData& NumItemsInHashTableSize (size_t num_items_in_hash_table_size)
    {
        num_items_in_hash_table_size_ = num_items_in_hash_table_size;
        return *this;
    }

    MatchingPerfData& TopksSize (size_t topks_size)
    {
        topks_size_ = topks_size;
        return *this;
    }

    MatchingPerfData& HashTableSize (size_t hash_table_size)
    {
        hash_table_size_ = hash_table_size;
        return *this;
    }

    MatchingPerfData& ComprRatio (float compr_ratio)
    {
        compr_ratio_ = compr_ratio;
        return *this;
    }

protected:
    GPUGenie::COMPRESSION_TYPE compr_;
    float overall_time_;
    float query_compilation_time_;
    float preprocessing_time_;
    float query_transfer_time_;
    float data_transfer_time_;
    float constant_transfer_time_;
    float allocation_time_;
    float filling_time_;
    float matching_time_;
    float convert_time_;
    size_t inv_size_;
    size_t dims_size_;
    size_t hash_table_capacity_per_query_;
    size_t threshold_size_;
    size_t passcount_size_;
    size_t bitmap_size_;
    size_t num_items_in_hash_table_size_;
    size_t topks_size_;
    size_t hash_table_size_;
    float compr_ratio_;
};



class ScanPerfData
{
public:
    ScanPerfData() :
        array_size_(0),
        time_(0.0),
        throughput_(0.0) {};
    
    void WriteHeader(std::ofstream &ofs)
    {
        ofs << "array_size" << ","
            << "time" << ","
            << "throughput" << std::endl;
    }

    void WriteLine(std::ofstream &ofs)
    {
        ofs << array_size_ << ","
            << time_ << ","
            << throughput_ << std::endl;
    }

    ScanPerfData& ArraySize(size_t array_size)
    {
        array_size_ = array_size;
        return *this;
    }

    ScanPerfData& Time(float time)
    {
        time_ = time;
        return *this;
    }

    ScanPerfData& Throughput(float throughput)
    {
        throughput_ = throughput;
        return *this;
    }

protected:
    size_t array_size_;
    float time_;
    float throughput_;
};



class CodecPerfData
{
public:
    CodecPerfData() :
        array_size_(0),
        time_(0.0),
        throughput_(0.0) {};
    
    void WriteHeader(std::ofstream &ofs)
    {
        ofs << "codec" << ","
            << "array_size" << ","
            << "compr_size" << ","
            << "ratio" << ","
            << "time" << ","
            << "throughput" << std::endl;
    }

    void WriteLine(std::ofstream &ofs)
    {
        ofs << static_cast<int>(codec_) << ","
            << array_size_ << ","
            << compr_size_ << ","
            << compr_ratio_ << ","
            << time_ << ","
            << throughput_ << std::endl;
    }

    CodecPerfData& Codec(GPUGenie::COMPRESSION_TYPE codec)
    {
        codec_ = codec;
        return *this;
    }

    CodecPerfData& ArraySize(size_t array_size)
    {
        array_size_ = array_size;
        return *this;
    }

    CodecPerfData& ComprSize(size_t compr_size)
    {
        compr_size_ = compr_size;
        return *this;
    }

    CodecPerfData& ComprRatio(float compr_ratio)
    {
        compr_ratio_ = compr_ratio;
        return *this;
    }

    CodecPerfData& Time(float time)
    {
        time_ = time;
        return *this;
    }

    CodecPerfData& Throughput(float throughput)
    {
        throughput_ = throughput;
        return *this;
    }

protected:
    GPUGenie::COMPRESSION_TYPE codec_;
    size_t array_size_;
    size_t compr_size_;
    float compr_ratio_;
    float time_;
    float throughput_;
};



} // namespace util
 
} // namespace genie

#endif
