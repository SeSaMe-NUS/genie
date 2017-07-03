#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "serialization.h"

namespace boost
{
namespace serialization
{


// // explicit instantiation of serialization functions
// template void serialize<boost::archive::text_iarchive>(
//     boost::archive::text_iarchive &ar, GPUGenie::inv_table &table, const unsigned int);

// explicit instantiation of serialization functions
// template void load<boost::archive::text_iarchive>(
//     boost::archive::text_iarchive &ar, GPUGenie::inv_table &table, const unsigned int);

// // explicit instantiation of serialization functions
// template void save<boost::archive::text_oarchive>(
//     boost::archive::text_oarchive &ar, GPUGenie::inv_table &table, const unsigned int);

} // namespace serialization
} // namespace boost

