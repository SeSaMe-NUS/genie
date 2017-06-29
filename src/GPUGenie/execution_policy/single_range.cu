#include <GPUGenie/table/interface.h>
#include <GPUGenie/query/interface.h>
#include <GPUGenie/matching/interface.h>
#include "single_range.h"

using namespace std;
using namespace genie;

shared_ptr<GPUGenie::inv_table> genie::execution_policy::SingleRangeExecutionPolicy::LoadTable(TableData& table_data)
{
	dim_ = table_data.at(0).size();

	return table::BuildTable(table_data);
}

vector<GPUGenie::query> genie::execution_policy::SingleRangeExecutionPolicy::LoadQuery(shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data)
{
	return query::LoadQuery(table, query_data, query_range_, k_);
}

SearchResult genie::execution_policy::SingleRangeExecutionPolicy::KnnSearch(shared_ptr<GPUGenie::inv_table>& table,
		vector<GPUGenie::query>& queries)
{
	return matching::Match(table, queries, dim_, k_);
}

void genie::execution_policy::SingleRangeExecutionPolicy::SetQueryRange(uint32_t query_range)
{
	query_range_ = query_range;
}
