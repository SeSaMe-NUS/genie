#include <fstream>
#include <memory>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include <genie/exception/genie_errors.h>
#include <genie/table/inv_compr_table.h>
#include <genie/table/inv_table.h>
#include <genie/utility/Logger.h>

#include "serialization.h"

void
genie::util::SaveTable(const std::string &filename, const std::shared_ptr<const GPUGenie::inv_table> &table)
{
    // Cannot save more than one table using this function
    if (table->get_table_index() != 0 || table->get_total_num_of_table() != 1)
        throw GPUGenie::genie_error("Saving multiple tables not supported");

    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);

    boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    out.push(boost::iostreams::bzip2_compressor());
    out.push(ofs);

    boost::archive::binary_oarchive oa(ofs);
    // oa.register_type<GPUGenie::inv_compr_table>();
    oa << table.get();
}

std::shared_ptr<GPUGenie::inv_table>
genie::util::LoadTable(const std::string &filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good())
        throw GPUGenie::genie_error("Loading from file failed (fstream not good)");

    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::bzip2_decompressor());
    in.push(ifs);

    boost::archive::binary_iarchive ia(ifs);
    // ia.register_type<GPUGenie::inv_compr_table>();

    GPUGenie::inv_table *loaded_table = nullptr;
    ia >> loaded_table;
    return std::shared_ptr<GPUGenie::inv_table>(loaded_table);
}

// Macro BOOST_CLASS_VERSION() only works for version < 256. Use the following for higher versions:
// namespace boost { namespace serialization {
//     template<> struct version<gps_position> {BOOST_STATIC_CONSTANT(int, value = APP_INT_VERSION); };
// }}
BOOST_CLASS_VERSION(GPUGenie::inv_table, 0);
BOOST_CLASS_VERSION(GPUGenie::inv_compr_table, 0);


template <class Archive>
void GPUGenie::inv_table::load(Archive &ar, const unsigned int version)
{
    ar.register_type(static_cast<GPUGenie::inv_compr_table*>(nullptr));
    Logger::log(Logger::DEBUG, "Loading inv_table archive of version %d", version);
    _build_status = builded;
    // General structure
    ar >> _size;
    ar >> _dim_size;
    // Inverted Index
    ar >> _inv; 
    ar >> _inv_pos;
    ar >> _inv_index_map;
    ar >> inv_list_lowerbound;
    ar >> inv_list_upperbound;
    ar >> posting_list_size;
    // Subsequence related fields
    ar >> shift_bits_subsequence;
    ar >> min_value_sequence;
    ar >> max_value_sequence;
    ar >> gram_length_sequence;
    ar >> shift_bits_sequence;
    ar >> _distinct_map;

}

template <class Archive>
void GPUGenie::inv_table::save(Archive &ar, const unsigned int version) const
{
    ar.register_type(static_cast<GPUGenie::inv_compr_table*>(nullptr));
    Logger::log(Logger::DEBUG, "Saving inv_table archive of version %d", version);
    if (_build_status != builded)
        throw GPUGenie::genie_error("Cannot serialize inv::table that has not yet been built.");
    // General structure
    ar << _size;
    ar << _dim_size;
    // Inverted Index
    ar << _inv; 
    ar << _inv_pos;
    ar << _inv_index_map;
    ar << inv_list_lowerbound;
    ar << inv_list_upperbound;
    ar << posting_list_size;
    // Subsequence related fields
    ar << shift_bits_subsequence;
    ar << min_value_sequence;
    ar << max_value_sequence;
    ar << gram_length_sequence;
    ar << shift_bits_sequence;
    ar << _distinct_map;
}


template <class Archive>
void GPUGenie::inv_compr_table::load(Archive &ar, const unsigned int version)
{
    Logger::log(Logger::DEBUG, "Loading inv_compr_table archive of version %d", version);
    ar >> boost::serialization::base_object<inv_table>(*this);

    ar >> m_isCompressed;
    ar >> m_compression;
    ar >> m_uncompressedInvListsMaxLength;
    ar >> m_comprInv;
    ar >> m_comprInvPos;

}

template <class Archive>
void GPUGenie::inv_compr_table::save(Archive &ar, const unsigned int version) const
{
    Logger::log(Logger::DEBUG, "Saving inv_compr_table archive of version %d", version);
    ar << boost::serialization::base_object<inv_table>(*this);

    ar << m_isCompressed;
    ar << m_compression;
    ar << m_uncompressedInvListsMaxLength;
    ar << m_comprInv;
    ar << m_comprInvPos;
}

