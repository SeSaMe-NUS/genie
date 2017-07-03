#include <GPUGenie.h>
#include "io.h"

using namespace std;
using namespace genie;

TableData genie::ReadTableFromCsv(const string& filename)
{
	TableData table_data;
	GPUGenie::read_file(table_data, filename.c_str(), -1);

	return table_data;
}

shared_ptr<GPUGenie::inv_table> genie::ReadTableFromBinary(const string& filename)
{
	GPUGenie::inv_table* table_ptr;
	GPUGenie::inv_table::read(filename.c_str(), table_ptr);
	shared_ptr<GPUGenie::inv_table> table(table_ptr, [](GPUGenie::inv_table* p) { delete[] p; });

	return table;
}

QueryData genie::ReadQueryFromCsv(const string& filename, shared_ptr<ExecutionPolicy>& policy)
{
	QueryData query_data;
	GPUGenie::read_file(query_data, filename.c_str(), policy->GetNumOfQuery());

	return query_data;
}
