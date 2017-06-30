#include "interface.h"

using namespace std;
using namespace genie;

vector<GPUGenie::query> genie::query::LoadQuery(shared_ptr<GPUGenie::inv_table>& table,
		QueryData& query_data,
		uint32_t query_range,
		uint32_t k)
{
	// TODO: once whole implementation is here, remove config completely
	vector<GPUGenie::query> queries;

	GPUGenie::GPUGenie_Config config;
	config.query_points = &query_data;
	config.query_radius = query_range;
	config.num_of_topk = k;
	config.dim = table->m_size();

	GPUGenie::load_query_singlerange(table.get()[0], queries, config);

	return queries;
}
