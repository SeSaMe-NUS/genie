#include <genie/interface/interface.h>

#include "interface.h"

using namespace std;
using namespace genie;

shared_ptr<genie::table::inv_table> genie::table::BuildTable(const TableData& table_data)
{
	genie::original::GPUGenie_Config config;
	config.data_points = &const_cast<TableData&>(table_data);
	genie::table::inv_table* table_ptr = nullptr;
	genie::original::preprocess_for_knn_csv(config, table_ptr);
	// Note: force array deleter to be used
	shared_ptr<genie::table::inv_table> table(table_ptr, [](genie::table::inv_table* p) { delete[] p; });

	return table;
}
