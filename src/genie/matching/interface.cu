#include <vector>
#include "interface.h"

using namespace std;
using namespace genie;

SearchResult genie::matching::Match(const shared_ptr<const GPUGenie::inv_table>& table,
		const vector<GPUGenie::query>& queries,
		const uint32_t dim,
		const uint32_t k)
{
	GPUGenie::GPUGenie_Config config;
	config.hashtable_size = dim * k * 1.5;
	config.count_threshold = dim;

	vector<int> topk, topk_count;
	GPUGenie::knn_search(const_cast<GPUGenie::inv_table&>(table.get()[0]),
			const_cast<vector<GPUGenie::query>&>(queries),
			const_cast<vector<int>&>(topk),
			const_cast<vector<int>&>(topk_count),
			const_cast<GPUGenie::GPUGenie_Config&>(config));

	return std::make_pair(move(topk), move(topk_count));
}
