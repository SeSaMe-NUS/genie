#include <GPUGenie/table/interface.h>
#include <GPUGenie/query/interface.h>
#include <GPUGenie/matching/interface.h>
#include "single_value.h"

using namespace std;
using namespace genie;

shared_ptr<GPUGenie::inv_table> genie::execution_policy::SingleValueExecutionPolicy::LoadTable(TableData& table_data)
{
	dim_ = table_data.at(0).size();

	return table::BuildTable(table_data);
}

vector<GPUGenie::query> genie::execution_policy::SingleValueExecutionPolicy::LoadQuery(shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data)
{
	return query::LoadQuery(table, query_data, 0, k_);
}

SearchResult genie::execution_policy::SingleValueExecutionPolicy::KnnSearch(shared_ptr<GPUGenie::inv_table>& table,
		vector<GPUGenie::query>& queries)
{
	return matching::Match(table, queries, dim_, k_);
}
