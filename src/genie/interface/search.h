#ifndef GENIE_INTERFACE_SEARCH_H_
#define GENIE_INTERFACE_SEARCH_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <genie/GPUGenie.h>
#include "types.h"

namespace genie {

/*!
 * \brief 1st-level interface for end-to-end matching with given table and query CSV file paths.
 */
SearchResult Search(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::string& table_filename,
		const std::string& query_filename);

/*!
 * \brief 2nd-level interface for building table with TableData.
 */
std::shared_ptr<genie::table::inv_table> BuildTable(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const TableData& table_data);

/*!
 * \brief 2nd-level interface for building queries with table and QueryData.
 */
std::vector<genie::query::Query> BuildQuery(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::shared_ptr<genie::table::inv_table>& table,
		const QueryData& query_data);

/*!
 * \brief 2nd-level interface for matching with pre-built table and queries.
 */
SearchResult Match(std::shared_ptr<genie::ExecutionPolicy>& policy,
		const std::shared_ptr<genie::table::inv_table>& table,
		const std::vector<genie::query::Query>& queries);

} // end of namespace genie

#endif
