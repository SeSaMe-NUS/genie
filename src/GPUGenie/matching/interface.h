#ifndef GENIE_MATCHING_INTERFACE_H_
#define GENIE_MATCHING_INTERFACE_H_

#include <memory>
#include <GPUGenie.h>
#include <GPUGenie/interface/types.h>

namespace genie {
namespace matching {

SearchResult Match(std::shared_ptr<GPUGenie::inv_table>& table,
		vector<GPUGenie::query>& queries,
		uint32_t dim,
		uint32_t k);

}
}

#endif
