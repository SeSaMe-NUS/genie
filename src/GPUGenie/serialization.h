#ifndef GENIE_SERIALIZATION_H_
#define GENIE_SERIALIZATION_H_

#include <iostream>
#include <typeinfo>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_free.hpp>

#include "inv_table.h"
#include "inv_compr_table.h"

// template <class Archive>
// void load(Archive &ar, GPUGenie::inv_table &table, const unsigned int);

// template <class Archive>
// void save(Archive &ar, GPUGenie::inv_table &table, const unsigned int);

namespace boost
{
namespace serialization
{

template <class Archive>
void load(Archive &ar, GPUGenie::inv_table &table, const unsigned int version)
{
    std::cout << "load(GPUGenie::inv_table)" << std::endl;
    ar >> table.table_index;

    try
    {
        GPUGenie::inv_compr_table &ctable = dynamic_cast<GPUGenie::inv_compr_table&>(table);
        ar >> ctable.m_compression;

    } catch (const std::bad_cast& e) {}

}

template <class Archive>
void save(Archive &ar, const GPUGenie::inv_table &table, const unsigned int version)
{
    std::cout << "save(GPUGenie::inv_table)" << std::endl;
    ar << table.table_index;

    try
    {
        const GPUGenie::inv_compr_table &ctable = dynamic_cast<const GPUGenie::inv_compr_table&>(table);
        ar << ctable.m_compression;

    } catch (const std::bad_cast& e) {}
}

// template <class Archive>
// void load(Archive &ar, GPUGenie::inv_compr_table &table, const unsigned int version)
// {
//     std::cout << "load(GPUGenie::inv_compr_table)" << std::endl;
//     ar >> boost::serialization::base_object<GPUGenie::inv_table>(table);
//     ar >> table.m_compression;
// }

// template <class Archive>
// void save(Archive &ar, const GPUGenie::inv_compr_table &table, const unsigned int version)
// {
//     std::cout << "save(GPUGenie::inv_compr_table)" << std::endl;
//     ar << boost::serialization::base_object<GPUGenie::inv_table>(table);
//     ar << table.m_compression;
// }

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

BOOST_CLASS_VERSION(GPUGenie::inv_table, 0);

#endif
