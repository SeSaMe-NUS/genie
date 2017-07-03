#include <iostream>
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
	TableData table_data = ReadTableFromCsv(table_filename);
	QueryData query_data = ReadQueryFromCsv(query_filename, policy);

	shared_ptr<GPUGenie::inv_table> table = LoadTable(policy, table_data);
	vector<GPUGenie::query> queries = LoadQuery(policy, table, query_data);

	return KnnSearch(policy, table, queries);
}

shared_ptr<GPUGenie::inv_table> genie::LoadTable(shared_ptr<genie::ExecutionPolicy>& policy,
		TableData& table_data)
{
	return policy->LoadTable(table_data);
}

vector<GPUGenie::query> genie::LoadQuery(shared_ptr<genie::ExecutionPolicy>& policy,
		shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data)
{
	return policy->LoadQuery(table, query_data);
}

SearchResult genie::KnnSearch(shared_ptr<genie::ExecutionPolicy>& policy,
		shared_ptr<GPUGenie::inv_table>& table,
		vector<GPUGenie::query>& queries)
{
	return policy->KnnSearch(table, queries);
}
