#ifndef KNN_H_
#define KNN_H_

#include <vector>
#include <thrust/device_vector.h>

#include "inv_table.h"
#include "query.h"

namespace GPUGenie
{
/**
 * @brief Find the top k values in given inv_table.
 * @details Find the top k values in given inv_table.
 *
 * @param table The given table.
 * @param queries The queries.
 * @param d_top_indexes The results' indexes.
 */
void
knn(inv_table& table, vector<query>& queries, device_vector<int>& d_top_indexes,
		device_vector<int>& d_top_count, int hash_table_size, int max_load,
		int bitmap_bits, int dim);
void
knn_bijectMap(inv_table& table, vector<query>& queries,
		device_vector<int>& d_top_indexes, device_vector<int>& d_top_count,
		int hash_table_size, int max_load, int bitmap_bits);
}

#endif //ifndef KNN_H_
