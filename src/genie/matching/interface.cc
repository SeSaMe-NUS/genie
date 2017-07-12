#include <vector>

#include <genie/original/interface.h>

#include "interface.h"

using namespace std;

genie::SearchResult genie::matching::Match(const shared_ptr<const genie::table::inv_table>& table,
		const vector<genie::query::Query>& queries,
		const uint32_t dim,
		const uint32_t k)
{
	genie::original::GPUGenie_Config config;
	config.hashtable_size = dim * k * 1.5;
	config.count_threshold = dim;

	vector<int> topk, topk_count;
	genie::original::knn_search(const_cast<genie::table::inv_table&>(table.get()[0]),
			const_cast<vector<genie::query::Query>&>(queries),
			const_cast<vector<int>&>(topk),
			const_cast<vector<int>&>(topk_count),
			const_cast<genie::original::GPUGenie_Config&>(config));

	return std::make_pair(move(topk), move(topk_count));
}
