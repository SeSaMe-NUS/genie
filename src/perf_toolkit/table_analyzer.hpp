
#ifndef INV_TABLE_ANALYZER_H_
#define INV_TABLE_ANALYZER_H_

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <genie/table/inv_table.h>
#include <genie/table/inv_compr_table.h>
#include <genie/compression/DeviceCodecFactory.h>
#include <genie/utility/PerfLogger.h>

namespace perftoolkit
{

class InvertedListData
{
public:
    InvertedListData() :
        compr_(genie::compression::NO_COMPRESSION),
        length_(0),
        raw_size_(0),
        compr_size_(0),
        low_value_(0),
        high_value_(0) {};
    
    void WriteHeader(std::ofstream &ofs)
    {
        ofs << "compr" << ","
            << "length" << ","
            << "raw_size" << ","
            << "compr_size" << ","
            << "low_value" << ","
            << "high_value" << std::endl;
    }

    void WriteLine(std::ofstream &ofs)
    {
        ofs << static_cast<int>(compr_) << ","
            << length_ << ","
            << raw_size_ << ","
            << compr_size_ << ","
            << low_value_ << ","
            << high_value_ << std::endl;
    }

    InvertedListData& Compr(genie::compression::COMPRESSION_TYPE compr)
    {
        compr_ = compr;
        return *this;
    }

    InvertedListData& Length(size_t length)
    {
        length_ = length;
        return *this;
    }

    InvertedListData& RawSize(size_t raw_size)
    {
        raw_size_ = raw_size;
        return *this;
    }

    InvertedListData& ComprSize(size_t compr_size)
    {
        compr_size_ = compr_size;
        return *this;
    }

    InvertedListData& FirstValue(int low_value)
    {
        low_value_ = low_value;
        return *this;
    }

    InvertedListData& LastValue(int high_value)
    {
        high_value_ = high_value;
        return *this;
    }

protected:
    genie::compression::COMPRESSION_TYPE compr_;
    size_t length_;
    size_t raw_size_;
    size_t compr_size_;
    int low_value_;
    int high_value_;
};


class InvertedListDistributionData
{
public:
    InvertedListDistributionData() :
        list_id_(-1) {};
    
    void WriteHeader(std::ofstream &ofs)
    {
        ofs << "list_id" << ","
            << "delta" << std::endl;
    }

    void WriteLine(std::ofstream &ofs)
    {
        ofs << list_id_ << ","
            << delta_ << std::endl;
    }

    InvertedListDistributionData& ListId(int list_id)
    {
        list_id_ = list_id;
        return *this;
    }

    InvertedListDistributionData& Delta(size_t delta)
    {
        delta_ = delta;
        return *this;
    }

protected:
    int list_id_;
    size_t delta_;
};


class TableAnalyzer
{
public:
    static void Analyze(std::shared_ptr<genie::table::inv_table> table, const std::string &dest_directory)
    {
        assert(table);
        assert(table->build_status() == genie::table::inv_compr_table::builded);

        genie::compression::COMPRESSION_TYPE compression = genie::compression::NO_COMPRESSION;
        std::vector<int> *inv;
        std::vector<int> *invPos;
        std::vector<int> *compressedInvPos;

        std::shared_ptr<genie::table::inv_compr_table> ctable =
                std::dynamic_pointer_cast<genie::table::inv_compr_table>(table);
        if (ctable)
        {
            compression = ctable->getCompression();
            inv = ctable->uncompressedInv();
            invPos = ctable->uncompressedInvPos();
            compressedInvPos = ctable->compressedInvPos();
        }
        else
        {
            inv = table->inv();
            invPos = table->inv_pos();
            compressedInvPos = table->inv_pos();
        }

        assert(invPos->size() == compressedInvPos->size());

        std::string dirsep("/");
        std::string fname(dest_directory+dirsep+"table_"
            +genie::compression::DeviceCodecFactory::getCompressionName(compression)+"_lists.csv");
        genie::utility::PerfLogger<InvertedListData>::Instance().New(fname);

        for (int pos = 0; pos < (int)invPos->size()-1; pos++)
        {
            genie::utility::PerfLogger<InvertedListData>::Instance().Log()
                .Compr(compression)
                .Length((*invPos)[pos+1]-(*invPos)[pos])
                .RawSize(((*invPos)[pos+1]-(*invPos)[pos]) * sizeof(int))
                .ComprSize(((*compressedInvPos)[pos+1]-(*compressedInvPos)[pos]) * sizeof(uint32_t))
                .FirstValue((*inv)[((*invPos)[pos])])
                .LastValue((*inv)[((*invPos)[pos+1]-1)]);

            genie::utility::PerfLogger<InvertedListData>::Instance().Next();
        }
    }
};

} // namespace perftoolkit
 
#endif
