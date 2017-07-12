#include <genie/table/interface.h>
#include <genie/query/interface.h>
#include <genie/matching/interface.h>
#include <genie/exception/exception.h>
#include "validation.h"
#include "single_range.h"

using namespace std;
using namespace genie;

shared_ptr<genie::table::inv_table> genie::execution_policy::SingleRangeExecutionPolicy::BuildTable(const TableData& table_data)
{
	dim_ = table_data.at(0).size();

	return table::BuildTable(table_data);
}

vector<genie::query::Query> genie::execution_policy::SingleRangeExecutionPolicy::BuildQuery(const shared_ptr<const genie::table::inv_table>& table,
		const QueryData& query_data)
{
	return query::LoadQuery(table, query_data, query_range_, k_);
}

SearchResult genie::execution_policy::SingleRangeExecutionPolicy::Match(const shared_ptr<const genie::table::inv_table>& table,
		const vector<genie::query::Query>& queries)
{
	return matching::Match(table, queries, dim_, k_);
}

SearchResult genie::execution_policy::SingleRangeExecutionPolicy::Match(const vector<shared_ptr<const genie::table::inv_table> >& table,
				const vector<vector<genie::query::Query> >& queries)
{
	throw genie::exception::NotImplementedException();
}

void genie::execution_policy::SingleRangeExecutionPolicy::SetQueryRange(const uint32_t query_range)
{
	query_range_ = query_range;
}

void genie::execution_policy::SingleRangeExecutionPolicy::Validate()
{
	ExecutionPolicy::Validate();
	execution_policy::validation::ValidateQueryRange(query_range_);
}
