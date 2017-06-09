
#ifndef INV_TABLE_ANALYZER_H_
#define INV_TABLE_ANALYZER_H_

#include <cassert>
#include <cstdint>
#include <fstream>
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

    virtual bool New(const std::string &filename)
    {
        ofs_.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
        if (ofs_.good())
            perf_data_.WriteHeader(ofs_);
        return ofs_.good();
    }

    virtual T& Set()
    {
        return perf_data_;
    }

    virtual void Next()
    {
        perf_data_.WriteLine(ofs_);
        perf_data_.Clear();
    }

    virtual ~PerfLogger()
    {
        Next();
        ofs_.close();
    }

protected:
    PerfLogger(){};
    PerfLogger(PerfLogger const&);
    PerfLogger(PerfLogger&&);
    PerfLogger<T>& operator=(PerfLogger const&);

    std::ofstream ofs_;
    T perf_data_;
};


class PerfData
{
public:
    virtual void WriteHeader(std::ofstream &ofs) = 0;
    virtual void WriteLine(std::ofstream &ofs) = 0;
    virtual void Clear() = 0;
};

template<typename T>
class TablePerfData : PerfData {};

template<>
class TablePerfData<GPUGenie::inv_table>
{
public:
    virtual void WriteHeader(std::ofstream &ofs) {assert(false);}
    virtual void WriteLine(std::ofstream &ofs) {assert(false);}
    virtual void Clear() {assert(false);}
    virtual TablePerfData<GPUGenie::inv_table>& InvListCount(size_t inv_lists_count)
    {
        inv_lists_count_ = inv_lists_count_;
        return *this;
    }
    virtual TablePerfData<GPUGenie::inv_table>& NumberOfInvertedLists(size_t number_of_inverted_lists)
    {
        number_of_inverted_lists_ = number_of_inverted_lists;
        return *this;
    }
    virtual TablePerfData<GPUGenie::inv_table>& InvListCount(const std::vector<size_t> &lists_raw_length)
    {
        lists_raw_length_ = lists_raw_length;
        return *this;
    }

protected:
    size_t inv_lists_count_;
    size_t number_of_inverted_lists_;
    std::vector<size_t> lists_raw_length_;
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


void testTableAnalyzer()
{
    using PerfLogger_t = PerfLogger<TablePerfData<GPUGenie::inv_table>>;
    PerfLogger_t::Instance().New("../results/inv-compr-table-stats.csv"); // write csv header
    PerfLogger_t::Instance().Set().InvListCount(9).NumberOfInvertedLists(23124);
    PerfLogger_t::Instance().Next(); // write csv line
}

} // namespace PerfToolkit
 
} // namespace GENIE

#endif
