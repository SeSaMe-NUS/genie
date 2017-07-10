#include <GPUGenie.h>
#include <GPUGenie/serialization.h>

#include "io.h"

using namespace std;
using namespace genie;

TableData genie::LoadTableDataFromCsv(const string& filename)
{
	TableData table_data;
	GPUGenie::read_file(table_data, filename.c_str(), -1);

	return table_data;
}

shared_ptr<GPUGenie::inv_table> genie::LoadTableFromBinary(const string& filename)
{
	return util::LoadTable(filename);;
}

void genie::SaveTableToBinary(const string& filename, const shared_ptr<const GPUGenie::inv_table> &table)
{
	util::SaveTable(filename, table);
}

QueryData genie::LoadQueryDataFromCsv(const string& filename, const shared_ptr<const ExecutionPolicy>& policy)
{
	QueryData query_data;
	GPUGenie::read_file(query_data, filename.c_str(), policy->GetNumOfQueries());

	return query_data;
}


