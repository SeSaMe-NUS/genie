#ifndef GENIE_SERIALIZATION_H_
#define GENIE_SERIALIZATION_H_

// #include <boost/serialization/split_free.hpp>

#include "inv_table.h"

// template <class Archive>
// void load(Archive &ar, GPUGenie::inv_table &table, const unsigned int);

// template <class Archive>
// void save(Archive &ar, GPUGenie::inv_table &table, const unsigned int);

// template <class Archive>
// void load(Archive &ar, GPUGenie::inv_table &table, const unsigned int)
// {
//     ar >> table.table_index;
// }

// template <class Archive>
// void save(Archive &ar, GPUGenie::inv_table &table, const unsigned int)
// {
//     ar << table.table_index;
// }


template <class Archive>
void serialize(Archive &ar, GPUGenie::inv_table &table, const unsigned int)
{
    ar & table.table_index;
}


// Macro BOOST_SERIALIZATION_SPLIT_FREE(GPUGenie::inv_table) expands to:
// namespace boost { namespace serialization {
// template<class Archive>
// inline void serialize(Archive & ar, GPUGenie::inv_table & t, const unsigned int file_version )
// {
//     split_free(ar, t, file_version); 
// }}}
// BOOST_SERIALIZATION_SPLIT_FREE(GPUGenie::inv_table)

#endif
