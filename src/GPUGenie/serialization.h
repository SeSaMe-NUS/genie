#ifndef GENIE_SERIALIZATION_H_
#define GENIE_SERIALIZATION_H_

#include <iostream>
#include <typeinfo>
#include <string>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include "genie_errors.h"
#include "inv_table.h"
#include "inv_compr_table.h"
#include "Logger.h"

namespace genie
{
namespace util
{

void
SaveTable(const std::string &filename, const GPUGenie::inv_table* table);

std::shared_ptr<GPUGenie::inv_table>
LoadTable(const std::string &filename);


}
}

namespace boost
{
namespace serialization
{

template <class Archive>
void load(Archive &ar, GPUGenie::inv_table &table, const unsigned int version)
{
    Logger::log(Logger::DEBUG, "Loading inv_table archive of version %d", version);
    ar >> table.table_index;
    ar >> table.total_num_of_table;
    ar >> table._shifter;
    ar >> table._size;
    ar >> table._dim_size;
    ar >> table.shift_bits_subsequence;
    ar >> table.min_value_sequence;
    ar >> table.max_value_sequence;
    ar >> table.gram_length_sequence;
    ar >> table.shift_bits_sequence;
    ar >> table._build_status;
    ar >> table._inv;
    ar >> table._inv_pos;
    ar >> table.inv_list_lowerbound;
    ar >> table.inv_list_upperbound;
    ar >> table.posting_list_size;
    ar >> table._inv_index;
    ar >> table._inv_index_map;
    ar >> table._distinct_map;

    bool isCompressed;
    ar >> isCompressed;
    if (isCompressed)
    {
        try
        {
            Logger::log(Logger::DEBUG, "Loading inv_compr_table archive of version %d", version);
            GPUGenie::inv_compr_table &ctable = dynamic_cast<GPUGenie::inv_compr_table&>(table);
            ar >> ctable.m_isCompressed;
            ar >> ctable.m_compression;
            ar >> ctable.m_uncompressedInvListsMaxLength;
            ar >> ctable.m_comprInv;
            ar >> ctable.m_comprInvPos;r

        }
        catch (const std::bad_cast& e)
        {
            throw GPUGenie::genie_error("Loading inverted table: archive says table should be compressed");
        }
    {
         = false;
        ar >>
    }

}

template <class Archive>
void save(Archive &ar, const GPUGenie::inv_table &table, const unsigned int version)
{
    Logger::log(Logger::DEBUG, "Saving inv_table archive of version %d", version);
    ar << table.table_index;
    ar << table.total_num_of_table;
    ar << table._shifter;
    ar << table._size;
    ar << table._dim_size;
    ar << table.shift_bits_subsequence;
    ar << table.min_value_sequence;
    ar << table.max_value_sequence;
    ar << table.gram_length_sequence;
    ar << table.shift_bits_sequence;
    ar << table._build_status;
    ar << table._inv;
    ar << table._inv_pos;
    ar << table.inv_list_lowerbound;
    ar << table.inv_list_upperbound;
    ar << table.posting_list_size;
    ar << table._inv_index;
    ar << table._inv_index_map;
    ar << table._distinct_map;

    try
    {
        Logger::log(Logger::DEBUG, "Saving inv_compr_table archive of version %d", version);
        const GPUGenie::inv_compr_table &ctable = dynamic_cast<const GPUGenie::inv_compr_table&>(table);
        ar << ctable.m_isCompressed;
        ar << ctable.m_compression;
        ar << ctable.m_uncompressedInvListsMaxLength;
        ar << ctable.m_comprInv;
        ar << ctable.m_comprInvPos;

    } catch (const std::bad_cast& e) {}
}


} // namespace serialization
} // nemaspace boosts

// Macro BOOST_SERIALIZATION_SPLIT_FREE(GPUGenie::inv_table) expands to:
// namespace boost { namespace serialization {
// template<class Archive>
// inline void serialize(Archive & ar, GPUGenie::inv_table & t, const unsigned int file_version )
// {
//     split_free(ar, t, file_version); 
// }}}
BOOST_SERIALIZATION_SPLIT_FREE(GPUGenie::inv_table);

// Macro BOOST_CLASS_VERSION() only works for version < 256. Use the following for higher versions:
// namespace boost { namespace serialization {
//     template<> struct version<gps_position> {BOOST_STATIC_CONSTANT(int, value = APP_INT_VERSION); };
// }}
BOOST_CLASS_VERSION(GPUGenie::inv_table, 0);

#endif
