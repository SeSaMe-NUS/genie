#include <iostream>
#include <GPUGenie/exception/exception.h>
#include <GPUGenie/interface/types.h>
#include <GPUGenie/interface/execution_policy.h>
#include <GPUGenie/interface/io.h>
#include <GPUGenie/interface/search.h>

using namespace std;
using namespace genie;

SearchResult genie::Search(shared_ptr<ExecutionPolicy>& policy,
		const string& table_filename,
		const string& query_filename)
{
	TableData table_data = LoadTableDataFromCsv(table_filename);
	QueryData query_data = LoadQueryDataFromCsv(query_filename, policy);

	shared_ptr<GPUGenie::inv_table> table = BuildTable(policy, table_data);
	vector<GPUGenie::query> queries = BuildQuery(policy, table, query_data);

	return Match(policy, table, queries);
}

shared_ptr<GPUGenie::inv_table> genie::BuildTable(shared_ptr<genie::ExecutionPolicy>& policy,
		const TableData& table_data)
{
	return policy->BuildTable(table_data);
}

vector<GPUGenie::query> genie::BuildQuery(shared_ptr<genie::ExecutionPolicy>& policy,
		const shared_ptr<GPUGenie::inv_table>& table,
		const QueryData& query_data)
{
	return policy->BuildQuery(table, query_data);
}

SearchResult genie::Match(shared_ptr<genie::ExecutionPolicy>& policy,
		const shared_ptr<GPUGenie::inv_table>& table,
		const vector<GPUGenie::query>& queries)
{
	return policy->Match(table, queries);
}
