#include "interface.h"

using namespace std;
using namespace genie;

vector<GPUGenie::query> genie::query::LoadQuery(const shared_ptr<const GPUGenie::inv_table>& table,
		const QueryData& query_data,
		const uint32_t query_range,
		const uint32_t k)
{
	// TODO: once whole implementation is here, remove config completely
	vector<GPUGenie::query> queries;

	GPUGenie::GPUGenie_Config config;
	config.query_points = &const_cast<QueryData&>(query_data);
	config.query_radius = query_range;
	config.num_of_topk = k;
	config.dim = const_cast<GPUGenie::inv_table*>(table.get())->m_size();

	GPUGenie::load_query_singlerange(const_cast<GPUGenie::inv_table&>(table.get()[0]), queries, config);

	return queries;
}
