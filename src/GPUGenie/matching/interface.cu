#include <vector>
#include "interface.h"

using namespace std;
using namespace genie;

SearchResult genie::matching::Match(shared_ptr<GPUGenie::inv_table>& table,
		vector<GPUGenie::query>& queries,
		uint32_t dim,
		uint32_t k)
{
	GPUGenie::GPUGenie_Config config;
	config.hashtable_size = dim * k * 1.5;
	config.count_threshold = dim;

	vector<int> topk, topk_count;
	GPUGenie::knn_search(table.get()[0], queries, topk, topk_count, config);

	return std::make_pair(move(topk), move(topk_count));
}
