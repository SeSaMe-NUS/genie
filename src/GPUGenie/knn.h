#ifndef KNN_H_
#define KNN_H_

#include "inv_list.h"
#include "inv_table.h"
#include "match.h"
#include "topk.h"
#include "query.h"
#include "raw_data.h"

namespace GPUGenie{
	/**
	 * @brief Find the top k values in given inv_table.
	 * @details Find the top k values in given inv_table.
	 *
	 * @param table The given table.
	 * @param queries The queries.
	 * @param d_top_indexes The results' indexes.
	 */
	void
	knn(inv_table& table,
		vector<query>& queries,
		device_vector<int>& d_top_indexes,
		device_vector<int>& d_top_count,
		int hash_table_size,
		int max_load,
		int bitmap_bits,
		int dim,
		int num_of_hot_dims,
		int hot_dim_threshold);
	void
	knn_bijectMap(inv_table& table,
		   vector<query>& queries,
		   device_vector<int>& d_top_indexes,
		   device_vector<int>& d_top_count,
		   int hash_table_size,
		   int max_load,
		   int bitmap_bits,
		   int dim,
		   int num_of_hot_dims,
		   int hot_dim_threshold);
}

#endif //ifndef KNN_H_
