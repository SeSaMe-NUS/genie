/*! \file knn.cu
 *  \brief Implementation for knn.h
 */
#include <math.h>
#include <assert.h>
#include <thrust/copy.h>

#include "inv_list.h"
#include "match.h"
#include "heap_count.h"

#include "Logger.h"
#include "Timing.h"
#include "genie_errors.h"

#include "knn.h"

bool GPUGENIE_ERROR = false;
unsigned long long GPUGENIE_TIME = 0ull;

#ifndef GPUGenie_knn_THREADS_PER_BLOCK
#define GPUGenie_knn_THREADS_PER_BLOCK 1024
#endif

#ifndef GPUGenie_knn_DEFAULT_HASH_TABLE_SIZE
#define GPUGenie_knn_DEFAULT_HASH_TABLE_SIZE 1
#endif

#ifndef GPUGenie_knn_DEFAULT_BITMAP_BITS
#define GPUGenie_knn_DEFAULT_BITMAP_BITS 2
#endif

#ifndef GPUGenie_knn_DEFAULT_DATA_PER_THREAD
#define GPUGenie_knn_DEFAULT_DATA_PER_THREAD 256
#endif

__global__
void extract_index_and_count(data_t * data, int * id, int * count, int size)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= size)
		return;
	id[tId] = data[tId].id;
	count[tId] = (int) data[tId].aggregation;
}

void GPUGenie::knn_bijectMap(GPUGenie::inv_table& table,
		vector<GPUGenie::query>& queries, device_vector<int>& d_top_indexes,
		device_vector<int>& d_top_count, int hash_table_size, int max_load,
		int bitmap_bits)
{
	int qmax = 0;

	for (unsigned int i = 0; i < queries.size(); ++i)
	{
		int count = queries[i].count_ranges();
		if (count > qmax)
			qmax = count;
	}

	u64 start = getTime();

	knn(table, queries, d_top_indexes, d_top_count, hash_table_size, max_load,
			bitmap_bits, float(qmax + 1));

	u64 end = getTime();
	double elapsed = getInterval(start, end);
	Logger::log(Logger::VERBOSE, ">>>>>>> knn takes %fms <<<<<<", elapsed);

}

void GPUGenie::knn_bijectMap_MT(vector<inv_table*>& table, vector<vector<query> >& queries,
		vector<device_vector<int> >& d_top_indexes, vector<device_vector<int> >& d_top_count,
		vector<int>& hash_table_size, vector<int>& max_load, int bitmap_bits)
{
	vector<int> qmaxs(table.size(), 0);

	auto it1 = qmaxs.begin();
	auto it2 = queries.begin();
	for (; it1 != qmaxs.end(); ++it1, ++it2)
	{
		for (auto it3 = it2->begin(); it3 != it2->end(); ++it3)
		{
			int count = it3->count_ranges();
			if (count > *it1)
				*it1 = count;
		}
	}
	
	u64 start = getTime();
	//knn(table, queries, d_top_indexes, d_top_count, hash_table_size, max_load, bitmap_bits, qmaxs);
	u64 end = getTime();

	double elapsed = getInterval(start, end);
	Logger::log(Logger::VERBOSE, ">>>>>>> knn takes %fms <<<<<<", elapsed);
}

void GPUGenie::knn(GPUGenie::inv_table& table, vector<GPUGenie::query>& queries,
		device_vector<int>& d_top_indexes, device_vector<int>& d_top_count,
		int hash_table_size, int max_load, int bitmap_bits, int dim)
{

	Logger::log(Logger::DEBUG, "Parameters: %d,%d,%d", hash_table_size,
			bitmap_bits, dim);
	dim = 2;

	device_vector<data_t> d_data;
	device_vector<u32> d_bitmap;

	device_vector<u32> d_num_of_items_in_hashtable(queries.size());

	device_vector<u32> d_threshold, d_passCount;

	Logger::log(Logger::DEBUG, "[knn] max_load is %d.", max_load);

	if(queries.empty()){
		throw GPUGenie::cpu_runtime_error("Queries not loaded!");
	}

	u64 startMatch = getTime();

	match(table, queries, d_data, d_bitmap, hash_table_size, max_load,
			bitmap_bits, d_num_of_items_in_hashtable, d_threshold, d_passCount);

	u64 endMatch = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>> match() takes %f ms <<<<<",
			getInterval(startMatch, endMatch));

	Logger::log(Logger::INFO, "Start topk....");
	u64 start = getTime();

	//topk(d_data, queries, d_top_indexes, float(dim));
	thrust::device_vector<data_t> d_topk;
	heap_count_topk(d_data, d_topk, d_threshold, d_passCount,
			queries[0].topk(),queries.size());

	u64 end = getTime();
	Logger::log(Logger::INFO, "Topk Finished!");
	Logger::log(Logger::VERBOSE, ">>>>> main topk takes %fms <<<<<",
			getInterval(start, end));
	start = getTime();


	d_top_count.resize(d_topk.size());
	d_top_indexes.resize(d_topk.size());
	extract_index_and_count<<<
			d_top_indexes.size() / GPUGenie_knn_THREADS_PER_BLOCK + 1,
			GPUGenie_knn_THREADS_PER_BLOCK>>>(
			thrust::raw_pointer_cast(d_topk.data()),
			thrust::raw_pointer_cast(d_top_indexes.data()),
			thrust::raw_pointer_cast(d_top_count.data()), d_top_indexes.size());


	end = getTime();
	Logger::log(Logger::INFO, "Finish topk search!");
	Logger::log(Logger::VERBOSE,
			">>>>> extract index and copy selected topk results takes %fms <<<<<",
			getInterval(start, end));
}
