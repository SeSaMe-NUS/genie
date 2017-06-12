
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
#include <GPUGenie/DeviceCodecFactory.h>
#include <GPUGenie/Logger.h>
#include <GPUGenie/PerfLogger.hpp>

using namespace GPUGenie;
using namespace genie::perf;

namespace genie
{

namespace perf_toolkit
{

class TableAnalyzer
{
public:
    static void Analyze(inv_table* table, const std::string &dest_directory)
    {
        assert(table);
        assert(table->build_status() == inv_compr_table::builded);

        COMPRESSION_TYPE compression = NO_COMPRESSION;
        std::vector<int> *inv;
        std::vector<int> *invPos;
        std::vector<int> *compressedInvPos;

        inv_compr_table *ctable = dynamic_cast<inv_compr_table*>(table);
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
        PerfLogger<InvertedListData>::Instance().New(
            dest_directory+dirsep+"table_"+DeviceCodecFactory::getCompressionName(compression)+"_lists.csv");

        for (int pos = 0; pos < (int)invPos->size()-1; pos++)
        {
            genie::perf::PerfLogger<genie::perf::InvertedListData>::Instance().Log()
                .Compr(compression)
                .Length((*invPos)[pos+1]-(*invPos)[pos])
                .RawSize(((*invPos)[pos+1]-(*invPos)[pos]) * sizeof(int))
                .ComprSize(((*compressedInvPos)[pos+1]-(*compressedInvPos)[pos]) * sizeof(uint32_t))
                .FirstValue((*inv)[((*invPos)[pos])])
                .LastValue((*inv)[((*invPos)[pos+1]-1)]);

            genie::perf::PerfLogger<genie::perf::InvertedListData>::Instance().Next();
        }
    }
};

} // namespace PerfToolkit
 
} // namespace GENIE

#endif
