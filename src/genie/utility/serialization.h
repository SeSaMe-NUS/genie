#ifndef GENIE_SERIALIZATION_H_
#define GENIE_SERIALIZATION_H_

#include <memory>
#include <string>

namespace genie
{
namespace utility
{

void
SaveTable(const std::string &filename, const std::shared_ptr<const genie::table::inv_table> &table);

std::shared_ptr<genie::table::inv_table>
LoadTable(const std::string &filename);

} // namespace utility
} // namesapce genie

#endif
