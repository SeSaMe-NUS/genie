#ifndef GENIE_INTERFACE_SEARCH_H_
#define GENIE_INTERFACE_SEARCH_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <GPUGenie.h>
#include "types.h"

namespace genie {

SearchResult Search(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::string& table_filename,
		const std::string& query_filename);

std::shared_ptr<GPUGenie::inv_table> LoadTable(std::shared_ptr<genie::ExecutionPolicy>& policy,
		TableData& table_data);

std::vector<GPUGenie::query> LoadQuery(std::shared_ptr<genie::ExecutionPolicy>& policy,
		std::shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data);

SearchResult KnnSearch(std::shared_ptr<genie::ExecutionPolicy>& policy,
		std::shared_ptr<GPUGenie::inv_table>& table,
		std::vector<GPUGenie::query>& queries);

} // end of namespace genie

#endif
