#ifndef GENIE_QUERY_INTERFACE_H_
#define GENIE_QUERY_INTERFACE_H_

#include <vector>
#include <GPUGenie.h>
#include <GPUGenie/interface/types.h>

namespace genie {
namespace query {

std::vector<GPUGenie::query> LoadQuery(std::shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data,
		uint32_t query_range,
		uint32_t k);

}
}

#endif
