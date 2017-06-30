#ifndef GENIE_TABLE_INTERFACE_H_
#define GENIE_TABLE_INTERFACE_H_

#include <memory>
#include <GPUGenie.h>
#include <GPUGenie/interface/types.h>

namespace genie {
namespace table {

std::shared_ptr<GPUGenie::inv_table> BuildTable(TableData& table_data);

}
}

#endif
