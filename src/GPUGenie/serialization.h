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

// Macro BOOST_CLASS_VERSION() only works for version < 256. Use the following for higher versions:
// namespace boost { namespace serialization {
//     template<> struct version<gps_position> {BOOST_STATIC_CONSTANT(int, value = APP_INT_VERSION); };
// }}
BOOST_CLASS_VERSION(GPUGenie::inv_table, 0);

#endif
