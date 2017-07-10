#ifndef GENIE_TABLE_INTERFACE_H_
#define GENIE_TABLE_INTERFACE_H_

#include <memory>
#include <genie/table/inv_table.h>
#include <genie/interface/types.h>

namespace genie {
namespace table {

std::shared_ptr<genie::table::inv_table> BuildTable(const TableData& table_data);

}
}

#endif
