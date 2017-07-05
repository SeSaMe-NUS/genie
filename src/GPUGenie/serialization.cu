#include <fstream>
#include <memory>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include "genie_errors.h"
#include "inv_compr_table.h"
#include "inv_table.h"
#include "Logger.h"

#include "serialization.h"

void
genie::util::SaveTable(const std::string &filename, const GPUGenie::inv_table* table)
{
    // Cannot save more than one table using this function
    if (table->get_table_index() != 0 || table->get_total_num_of_table() != 1)
        throw GPUGenie::genie_error("Saving multiple tables not supported");

    std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::trunc);
    boost::archive::text_oarchive oa(ofs);
    // oa.register_type<GPUGenie::inv_compr_table>();
    oa << table;
}

std::shared_ptr<GPUGenie::inv_table>
genie::util::LoadTable(const std::string &filename)
{
    GPUGenie::inv_table *loaded_table = nullptr;
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    boost::archive::text_iarchive ia(ifs);
    // ia.register_type<GPUGenie::inv_compr_table>();
    ia >> loaded_table;
    return shared_ptr<GPUGenie::inv_table>(loaded_table);
}

// Macro BOOST_CLASS_VERSION() only works for version < 256. Use the following for higher versions:
// namespace boost { namespace serialization {
//     template<> struct version<gps_position> {BOOST_STATIC_CONSTANT(int, value = APP_INT_VERSION); };
// }}
BOOST_CLASS_VERSION(GPUGenie::inv_table, 0);


template <class Archive>
void GPUGenie::inv_table::load(Archive &ar, const unsigned int version)
{
    ar.register_type(static_cast<GPUGenie::inv_compr_table*>(nullptr));
    Logger::log(Logger::DEBUG, "Loading inv_table archive of version %d", version);
    ar >> table_index;
    ar >> total_num_of_table;
    ar >> _shifter;
    ar >> _size;
    ar >> _dim_size;
    ar >> shift_bits_subsequence;
    ar >> min_value_sequence;
    ar >> max_value_sequence;
    ar >> gram_length_sequence;
    ar >> shift_bits_sequence;
    ar >> _build_status;
    ar >> _inv;
    ar >> _inv_pos;
    ar >> inv_list_lowerbound;
    ar >> inv_list_upperbound;
    ar >> posting_list_size;
    ar >> _inv_index;
    ar >> _inv_index_map;
    ar >> _distinct_map;

}

template <class Archive>
void GPUGenie::inv_table::save(Archive &ar, const unsigned int version) const
{
    ar.register_type(static_cast<GPUGenie::inv_compr_table*>(nullptr));
    Logger::log(Logger::DEBUG, "Saving inv_table archive of version %d", version);
    ar << table_index;
    ar << total_num_of_table;
    ar << _shifter;
    ar << _size;
    ar << _dim_size;
    ar << shift_bits_subsequence;
    ar << min_value_sequence;
    ar << max_value_sequence;
    ar << gram_length_sequence;
    ar << shift_bits_sequence;
    ar << _build_status;
    ar << _inv;
    ar << _inv_pos;
    ar << inv_list_lowerbound;
    ar << inv_list_upperbound;
    ar << posting_list_size;
    ar << _inv_index;
    ar << _inv_index_map;
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

