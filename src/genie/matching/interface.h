#ifndef GENIE_MATCHING_INTERFACE_H_
#define GENIE_MATCHING_INTERFACE_H_

#include <memory>
#include <vector>
#include <genie/query/query.h>
#include <genie/table/inv_table.h>
#include <genie/interface/types.h>

namespace genie {
namespace matching {

SearchResult Match(const std::shared_ptr<const genie::table::inv_table>& table,
		const std::vector<genie::query::Query>& queries,
		const uint32_t dim,
		const uint32_t k);

}
}

#endif
