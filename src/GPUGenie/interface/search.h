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

std::shared_ptr<GPUGenie::inv_table> BuildTable(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const TableData& table_data);

std::vector<GPUGenie::query> BuildQuery(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::shared_ptr<GPUGenie::inv_table>& table,
		const QueryData& query_data);

SearchResult Match(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::shared_ptr<GPUGenie::inv_table>& table,
		const std::vector<GPUGenie::query>& queries);

} // end of namespace genie

#endif
