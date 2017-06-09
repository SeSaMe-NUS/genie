
#ifndef INV_TABLE_ANALYZER_H_
#define INV_TABLE_ANALYZER_H_

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <GPUGenie/inv_table.h>
#include <GPUGenie/inv_compr_table.h>
#include <GPUGenie/Logger.h>

using namespace GPUGenie;

namespace genie
{

namespace perf
{

template <typename T>
class PerfLogger{
public:
    static PerfLogger& Instance()
    {
        static PerfLogger<T> pl;
        return pl;
    }

    bool New(const std::string &filename)
    {
        ofs_.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
        assert(ofs_.good());
        perf_data_->WriteHeader(ofs_);
        return ofs_.good();
    }

    T& Log()
    {
        assert(ofs_.good());
        dirty_perf_data_ = true;
        return *(perf_data_.get());
    }

    void Reset()
    {
        perf_data_.reset(new T);
        dirty_perf_data_ = false;
    }

    void Next()
    {
        assert(ofs_.good());
        assert(dirty_perf_data_);
        perf_data_->WriteLine(ofs_);
        Reset();
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

template<typename T>
class TablePerfData;

template<>
class TablePerfData<GPUGenie::inv_table>
{
public:
    void WriteHeader(std::ofstream &ofs) {assert(false);}
    
    void WriteLine(std::ofstream &ofs) {assert(false);}

    TablePerfData<GPUGenie::inv_table>& InvListCount(size_t inv_lists_count)
    {
        inv_lists_count_ = inv_lists_count_;
        return *this;
    }
    TablePerfData<GPUGenie::inv_table>& NumberOfInvertedLists(size_t number_of_inverted_lists)
    {
        number_of_inverted_lists_ = number_of_inverted_lists;
        return *this;
    }
    TablePerfData<GPUGenie::inv_table>& InvListCount(const std::vector<size_t> &lists_raw_length)
    {
        lists_raw_length_ = lists_raw_length;
        return *this;
    }

protected:
    size_t inv_lists_count_;
    size_t number_of_inverted_lists_;
    std::vector<size_t> lists_raw_length_;
};


class IntegratedMatchingPerfData
{
public:
    IntegratedMatchingPerfData() :
        compr_(NO_COMPRESSION),
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
        ofs << DeviceCodecFactory::getCompressionName(compr_) << ","
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

    IntegratedMatchingPerfData& Compr (COMPRESSION_TYPE compr)
    {
        compr_ = compr;
        return *this;
    }

    IntegratedMatchingPerfData& OverallTime (float overall_time)
    {
        overall_time_ = overall_time;
        return *this;
    }

    IntegratedMatchingPerfData& QueryCompilationTime (float query_compilation_time)
    {
        query_compilation_time_ = query_compilation_time;
        return *this;
    }

    IntegratedMatchingPerfData& PreprocessingTime (float preprocessing_time)
    {
        preprocessing_time_ = preprocessing_time;
        return *this;
    }

    IntegratedMatchingPerfData& QueryTransferTime (float query_transfer_time)
    {
        query_transfer_time_ = query_transfer_time;
        return *this;
    }

    IntegratedMatchingPerfData& DataTransferTime (float data_transfer_time)
    {
        data_transfer_time_ = data_transfer_time;
        return *this;
    }

    IntegratedMatchingPerfData& ConstantTransferTime (float constant_transfer_time)
    {
        constant_transfer_time_ = constant_transfer_time;
        return *this;
    }

    IntegratedMatchingPerfData& AllocationTime (float allocation_time)
    {
        allocation_time_ = allocation_time;
        return *this;
    }

    IntegratedMatchingPerfData& FillingTime (float filling_time)
    {
        filling_time_ = filling_time;
        return *this;
    }

    IntegratedMatchingPerfData& MatchingTime (float matching_time)
    {
        matching_time_ = matching_time;
        return *this;
    }

    IntegratedMatchingPerfData& ConvertTime (float convert_time)
    {
        convert_time_ = convert_time;
        return *this;
    }

    IntegratedMatchingPerfData& InvSize (size_t inv_size)
    {
        inv_size_ = inv_size;
        return *this;
    }

    IntegratedMatchingPerfData& DimsSize (size_t dims_size)
    {
        dims_size_ = dims_size;
        return *this;
    }

    IntegratedMatchingPerfData& HashTableCapacityPerQuery (size_t hash_table_capacity_per_query)
    {
        hash_table_capacity_per_query_ = hash_table_capacity_per_query;
        return *this;
    }

    IntegratedMatchingPerfData& ThresholdSize (size_t threshold_size)
    {
        threshold_size_ = threshold_size;
        return *this;
    }

    IntegratedMatchingPerfData& PasscountSize (size_t passcount_size)
    {
        passcount_size_ = passcount_size;
        return *this;
    }

    IntegratedMatchingPerfData& BitmapSize (size_t bitmap_size)
    {
        bitmap_size_ = bitmap_size;
        return *this;
    }

    IntegratedMatchingPerfData& NumItemsInHashTableSize (size_t num_items_in_hash_table_size)
    {
        num_items_in_hash_table_size_ = num_items_in_hash_table_size;
        return *this;
    }

    IntegratedMatchingPerfData& TopksSize (size_t topks_size)
    {
        topks_size_ = topks_size;
        return *this;
    }

    IntegratedMatchingPerfData& HashTableSize (size_t hash_table_size)
    {
        hash_table_size_ = hash_table_size;
        return *this;
    }

    IntegratedMatchingPerfData& ComprRatio (float compr_ratio)
    {
        compr_ratio_ = compr_ratio;
        return *this;
    }

protected:
    COMPRESSION_TYPE compr_;
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

// template<>
// class TablePerfData<GPUGenie::inv_compr_table> : TablePerfData<GPUGenie::inv_table>
// {
// public:
//     float compression_ratio;
//     size_t inv_lists_count;
//     size_t number_of_inverted_lists;
//     std::vector<float> lists_compression_ratio;
//     std::vector<size_t> lists_compressed_length;
//     std::vector<size_t> lists_raw_length;
// };

} // namespace PerfToolkit
 
} // namespace GENIE

#endif
