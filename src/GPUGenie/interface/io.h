#ifndef GENIE_INTERFACE_IO_H_
#define GENIE_INTERFACE_IO_H_

#include <vector>
#include <string>
#include <memory>
#include <GPUGenie.h>
#include <GPUGenie/interface/execution_policy.h>
#include "types.h"

namespace genie {

TableData ReadTableFromCsv(const std::string& filename);
std::shared_ptr<GPUGenie::inv_table> ReadTableFromBinary(const std::string& filename);
QueryData ReadQueryFromCsv(const std::string& filename, std::shared_ptr<ExecutionPolicy>& policy);

} // end of namespace genie

#endif
