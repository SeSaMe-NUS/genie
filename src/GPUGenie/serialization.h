#ifndef GENIE_SERIALIZATION_H_
#define GENIE_SERIALIZATION_H_

#include <memory>
#include <string>

namespace genie
{
namespace util
{

void
SaveTable(const std::string &filename, const std::shared_ptr<const GPUGenie::inv_table> &table);

std::shared_ptr<GPUGenie::inv_table>
LoadTable(const std::string &filename);

} // namespace util
} // namesapce genie

#endif
