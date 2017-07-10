#ifndef GENIE_QUERY_INTERFACE_H_
#define GENIE_QUERY_INTERFACE_H_

#include <vector>
#include <genie/GPUGenie.h>
#include <genie/interface/types.h>

namespace genie {
namespace query {

std::vector<GPUGenie::query> LoadQuery(const std::shared_ptr<const GPUGenie::inv_table>& table,
		const QueryData& query_data,
		const uint32_t query_range,
		const uint32_t k);

}
}

#endif
