#ifndef GENIE_QUERY_INTERFACE_H_
#define GENIE_QUERY_INTERFACE_H_

#include <memory>
#include <vector>
#include <genie/query/query.h>
#include <genie/table/inv_table.h>
#include <genie/interface/types.h>

namespace genie {
namespace query {

std::vector<genie::query::Query> LoadQuery(const std::shared_ptr<const genie::table::inv_table>& table,
		const QueryData& query_data,
		const uint32_t query_range,
		const uint32_t k);

}
}

#endif
