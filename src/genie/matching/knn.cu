/*! \file knn.cu
 *  \brief Implementation for knn.h
 */
#include <math.h>
#include <assert.h>

#include <genie/table/inv_list.h>
#include "match.h"
#include "match_integrated.h"
#include "heap_count.h"

#include <genie/utility/Logger.h>
#include <genie/utility/Timing.h>
#include <genie/utility/cuda_macros.h>
#include <genie/exception/exception.h>

#include <genie/configure.h>

#include "knn.h"

using namespace genie::table;
using namespace genie::query;
using namespace genie::matching;
using namespace genie::utility;
using namespace std;
using namespace thrust;

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

void genie::matching::knn_bijectMap(genie::table::inv_table& table,
		vector<Query>& queries, device_vector<int>& d_top_indexes,
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
			bitmap_bits);

	u64 end = getTime();
	double elapsed = getInterval(start, end);
	Logger::log(Logger::VERBOSE, ">>>>>>> knn takes %fms <<<<<<", elapsed);

}

void genie::matching::knn_bijectMap_MT(vector<inv_table*>& table, vector<vector<Query> >& queries,
		vector<device_vector<int> >& d_top_indexes, vector<device_vector<int> >& d_top_count,
		vector<int>& hash_table_size, vector<int>& max_load, int bitmap_bits)
{
	vector<int> qmaxs(table.size(), 0);

	auto it1 = qmaxs.begin();
	auto it2 = queries.begin();
	for (; it1 != qmaxs.end(); ++it1, ++it2)
		for (auto it3 = it2->begin(); it3 != it2->end(); ++it3)
		{
			int count = it3->count_ranges();
			if (count > *it1)
				*it1 = count;
		}
	
	u64 start = getTime();
	knn_MT(table, queries, d_top_indexes, d_top_count, hash_table_size, max_load, bitmap_bits);
	u64 end = getTime();

	double elapsed = getInterval(start, end);
	Logger::log(Logger::VERBOSE, ">>>>>>> knn takes %fms <<<<<<", elapsed);
}

void genie::matching::knn(genie::table::inv_table& table, vector<Query>& queries,
		device_vector<int>& d_top_indexes, device_vector<int>& d_top_count,
		int hash_table_size, int max_load, int bitmap_bits)
{
	Logger::log(Logger::DEBUG, "Parameters: %d,%d", hash_table_size, bitmap_bits);

	device_vector<data_t> d_data;
	device_vector<u32> d_bitmap;

	device_vector<u32> d_num_of_items_in_hashtable(queries.size());

	device_vector<u32> d_threshold, d_passCount;

	Logger::log(Logger::DEBUG, "[knn] max_load is %d.", max_load);

	if(queries.empty()){
		throw genie::exception::cpu_runtime_error("Queries not loaded!");
	}

	u64 startMatch = getTime();

	#ifdef GENIE_COMPR
	genie::table::inv_compr_table *comprTable = dynamic_cast<inv_compr_table*>(&table);
	if (comprTable){
		MatchIntegratedFunPtr matchFn = genie::compression::DeviceCodecFactory::getMatchingFunPtr(comprTable->getCompression());
		if (!matchFn)
		{
			Logger::log(Logger::ALERT, "No matching function for %s compression!",
				genie::compression::DeviceCodecFactory::getCompressionName(comprTable->getCompression()).c_str());
			throw genie::exception::cpu_runtime_error("No compression matching function avaiable for required compression!");
		}

		matchFn(
			*comprTable, queries, d_data, d_bitmap,
			hash_table_size, bitmap_bits, d_num_of_items_in_hashtable, d_threshold, d_passCount);

	} else
	#endif
	{
		match(table, queries, d_data, d_bitmap,
			hash_table_size, max_load, bitmap_bits, d_num_of_items_in_hashtable, d_threshold, d_passCount);
	}

	u64 endMatch = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>> match() takes %f ms <<<<<",
			getInterval(startMatch, endMatch));

	Logger::log(Logger::INFO, "Start topk....");
	u64 start = getTime();

	thrust::device_vector<data_t> d_topk;
	genie::matching::heap_count_topk(d_data, d_topk, d_threshold, d_passCount,
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

void
genie::matching::knn_MT(vector<inv_table*>& table, vector<vector<Query> >& queries,
		vector<device_vector<int> >& d_top_indexes, vector<device_vector<int> >& d_top_count,
		vector<int>& hash_table_size, vector<int>& max_load, int bitmap_bits)
{
	/* pre-process */
	vector<device_vector<data_t> > d_data(table.size());
	vector<device_vector<u32> >    d_bitmap(table.size());
	vector<device_vector<u32> >    d_num_of_items_in_hashtable(table.size());
	vector<device_vector<u32> >    d_threshold(table.size());
	vector<device_vector<u32> >    d_passCount(table.size());
	vector<device_vector<data_t> > d_topk(table.size());
	for (size_t i = 0; i < table.size(); ++i)
	{
		d_num_of_items_in_hashtable.at(i).resize(queries.at(i).size());
		Logger::log(Logger::DEBUG, "[knn] max_load is %d.", max_load.at(i));
		//if (queries.at(i).empty())
		//	clog << "No query on table " << i << "/" << table.size() << endl;
	}

	/* run batched match kernels */
	u64 startMatch = getTime();
	size_t query_bytesize, gpu_free_mem, gpu_total_mem;
	size_t start = 0, finish = 0;
	size_t tolerance = 50 * 1024 * 1024; // make sure GPU always has at least 50 MB left
	cudaCheckErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
	while (true)
	{
		/* if not all tables are already processed */
		if (table.size() != start)
		{
			/* if not all remaining tables are scheduled for processing */
			if (table.size() != finish)
			{
				// TODO: accurately estimate GPU memory size
				query_bytesize = queries.at(finish).size() * hash_table_size.at(finish) * sizeof(data_t) + // d_data
					queries.at(finish).size() * sizeof(u32) + // d_noiih
					queries.at(finish).size() * sizeof(u32) + // d_threshold
					queries.at(finish).size() * table.at(finish)->m_size() + // d_passCount
					queries.at(finish).size() * sizeof(u32) + // d_topk in match
					queries.at(finish).size() * table.at(finish)->m_size() * sizeof(Query::dim) + // d_dims
					queries.at(finish).size() * table.at(finish)->i_size(); // d_bitmap
				if (!queries.at(finish).empty())
					query_bytesize += queries.at(finish).size() * queries.at(finish).at(0).topk() * sizeof(data_t); // d_topk
				/* if mem has space */
				if (gpu_free_mem > query_bytesize + tolerance)
				{
					gpu_free_mem -= query_bytesize;
					++finish;
					continue;
				}
				/* cannot fit a single table */
				else if (start == finish)
					throw genie::exception::gpu_bad_alloc("MEMORY NOT ENOUGH");
			}
			/* match and extract top k for a batch */
			match_MT(table, queries, d_data, d_bitmap, hash_table_size, max_load,
					bitmap_bits, d_num_of_items_in_hashtable, d_threshold, d_passCount, start, finish);
			for (size_t i = start; i < finish; ++i)
			{
				if (queries.at(i).empty())
					continue;
				heap_count_topk(d_data.at(i), d_topk.at(i), d_threshold.at(i), d_passCount.at(i),
						queries.at(i).at(0).topk(), queries.at(i).size());

				d_top_count.at(i).resize(d_topk.at(i).size());
				d_top_indexes.at(i).resize(d_topk.at(i).size());
				extract_index_and_count<<<
						d_top_indexes.at(i).size() / GPUGenie_knn_THREADS_PER_BLOCK + 1,
						GPUGenie_knn_THREADS_PER_BLOCK>>>(
						thrust::raw_pointer_cast(d_topk.at(i).data()),
						thrust::raw_pointer_cast(d_top_indexes.at(i).data()),
						thrust::raw_pointer_cast(d_top_count.at(i).data()), d_top_indexes.at(i).size());

				d_data.at(i).clear();
				d_data.at(i).shrink_to_fit();
				d_bitmap.at(i).clear();
				d_bitmap.at(i).shrink_to_fit();
				d_topk.at(i).clear();
				d_topk.at(i).shrink_to_fit();
				d_num_of_items_in_hashtable.at(i).clear();
				d_num_of_items_in_hashtable.at(i).shrink_to_fit();
				d_threshold.at(i).clear();
				d_threshold.at(i).shrink_to_fit();
				d_passCount.at(i).clear();
				d_passCount.at(i).shrink_to_fit();
			}
			/* update mem info after memory free and continue with next batch */
			cudaCheckErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
			start = finish;
		}
		else
			break;
	}
	u64 endMatch = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>> match() takes %f ms <<<<<",
			getInterval(startMatch, endMatch));
}
