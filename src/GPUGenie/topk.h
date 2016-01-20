#ifndef GPUGenie_topk_h
#define GPUGenie_topk_h

#include <vector>
#include <thrust/device_vector.h>

#include "match.h"

using namespace std;
using namespace thrust;

namespace GPUGenie
{

int
calculate_bits_per_data(int bitmap_bits);
/**
 * @brief Find the top k values in given device_vector.
 * @details Find the top k values in given device_vector.
 *
 * @param d_search The data vector.
 * @param queries The queries.
 * @param d_top_indexes The results' indexes.
 */
void
topk(device_vector<int>& d_search, vector<query>& queries,
		device_vector<int>& d_top_indexes);

/**
 * @brief Find the top k values in given device_vector.
 * @details Find the top k values in given device_vector.
 *
 * @param d_search The data vector.
 * @param queries The queries.
 * @param d_top_indexes The results' indexes.
 */
void
topk(device_vector<float>& d_search, vector<query>& queries,
		device_vector<int>& d_top_indexes);
void
topk(device_vector<data_t>& d_search, vector<query>& queries,
		device_vector<int>& d_top_indexe, float dim);
void
topk(device_vector<u32>& d_search, vector<GPUGenie::query>& queries,
		device_vector<int>& d_top_indexes, u32 dim);

/**
 * @brief Find the top k values in given device_vector.
 * @details Find the top k values in given device_vector.
 *
 * @param d_search The data vecto.
 * @param d_tops The top k values.
 * @param d_top_indexes The results' indexes.
 */
void
topk(device_vector<int>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes);

/**
 * @brief Find the top k values in given device_vector.
 * @details Find the top k values in given device_vector.
 *
 * @param d_search The data vector
 * @param d_tops The top k values.
 * @param d_top_indexes The results' indexes.
 */
void
topk(device_vector<float>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes);
void
topk(device_vector<data_t>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes, float dim);
void
topk(device_vector<u32>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes, u32 dim);
}

#endif
