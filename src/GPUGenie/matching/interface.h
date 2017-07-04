#ifndef GENIE_MATCHING_INTERFACE_H_
#define GENIE_MATCHING_INTERFACE_H_

#include <memory>
#include <vector>
#include <GPUGenie.h>
#include <GPUGenie/interface/types.h>

namespace genie {
namespace matching {

SearchResult Match(const std::shared_ptr<const GPUGenie::inv_table>& table,
		const std::vector<GPUGenie::query>& queries,
		const uint32_t dim,
		const uint32_t k);

}
}

#endif
