#include <math.h>
#include <assert.h>
#include <thrust/copy.h>

#include "raw_data.h"
#include "inv_list.h"
#include "match.h"
#include "topk.h"

#include "Logger.h"
#include "Timing.h"

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
void extract_index_and_count(int * id, int * count, data_t * od, int size)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= size)
		return;
	int topk_id = id[tId];
	id[tId] = od[topk_id].id;
	count[tId] = (int) od[topk_id].aggregation;
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
void GPUGenie::knn(GPUGenie::inv_table& table, vector<GPUGenie::query>& queries,
		device_vector<int>& d_top_indexes, device_vector<int>& d_top_count,
		int hash_table_size, int max_load, int bitmap_bits, int dim)
{

	Logger::log(Logger::DEBUG, "Parameters: %d,%d,%d", hash_table_size,
			bitmap_bits, dim);

	//for improve
//  int qmax = 0;
//  for(int i = 0; i < queries.size(); ++i)
//  {
//	 int count = queries[i].count_ranges();
//	  if(count > qmax)
//		  qmax = count;
//  }
	//end for improve
	dim = 2;

	u64 startKnn = getTime();

	device_vector<data_t> d_data;
	device_vector<u32> d_bitmap;

	u64 end2Knn = getTime();
	Logger::log(Logger::VERBOSE, ">>>>> knn() before match() %f ms <<<<<",
			getInterval(startKnn, end2Knn));

	device_vector<u32> d_num_of_items_in_hashtable(queries.size());

	Logger::log(Logger::DEBUG, "[knn] max_load is %d.", max_load);

	match(table, queries, d_data, d_bitmap, hash_table_size, max_load,
			bitmap_bits, d_num_of_items_in_hashtable);

	u64 end1Knn = getTime();
	Logger::log(Logger::VERBOSE, ">>>>> knn() after match() %f ms <<<<<",
			getInterval(startKnn, end1Knn));

	u64 endKnn = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>> knn() before topk and extractIndex %f ms <<<<<",
			getInterval(startKnn, endKnn));

	Logger::log(Logger::INFO, "Start topk....");
	u64 start = getTime();

	topk(d_data, queries, d_top_indexes, float(dim));

	u64 end = getTime();
	Logger::log(Logger::INFO, "Topk Finished!");
	Logger::log(Logger::VERBOSE, ">>>>> main topk takes %fms <<<<<",
			getInterval(start, end));
	start = getTime();

	d_top_count.resize(d_top_indexes.size());
	extract_index_and_count<<<
			d_top_indexes.size() / GPUGenie_knn_THREADS_PER_BLOCK + 1,
			GPUGenie_knn_THREADS_PER_BLOCK>>>(
			thrust::raw_pointer_cast(d_top_indexes.data()),
			thrust::raw_pointer_cast(d_top_count.data()),
			thrust::raw_pointer_cast(d_data.data()), d_top_indexes.size());

	end = getTime();
	Logger::log(Logger::INFO, "Finish topk search!");
	Logger::log(Logger::VERBOSE,
			">>>>> extract index and copy selected topk results takes %fms <<<<<",
			getInterval(start, end));

}
