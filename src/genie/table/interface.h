#ifndef GENIE_TABLE_INTERFACE_H_
#define GENIE_TABLE_INTERFACE_H_

#include <memory>
#include <genie/GPUGenie.h>
#include <genie/interface/types.h>

namespace genie {
namespace table {

std::shared_ptr<GPUGenie::inv_table> BuildTable(const TableData& table_data);

}
}

#endif
