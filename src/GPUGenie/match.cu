/*! \file match.cu
 *  \brief Implementation for match.h
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "Logger.h"
#include "PerfLogger.hpp"
#include "Timing.h"
#include "genie_errors.h"
#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"

#include "match.h"
#include "match_common.h"
#include "match_device_utils.h"

using namespace genie::matching;
using namespace genie::util;
using namespace std;
using namespace thrust;

namespace GPUGenie
{


//for AT: for adaptiveThreshold match function for adaptiveThreshold
__global__
void match_AT(int m_size, int i_size, int hash_table_size,
		int* d_inv, query::dim* d_dims,
		T_HASHTABLE* hash_table_list, u32 * bitmap_list, int bitmap_bits,
		u32* d_topks, u32* d_threshold, //initialized as 1, and increase gradually
		u32* d_passCount, //initialized as 0, count the number of items passing one d_threshold
		u32 num_of_max_count, u32 * noiih, bool * overflow, unsigned int shift_bits_subsequence)
{
	if (m_size == 0 || i_size == 0)
		return;
	query::dim& q = d_dims[blockIdx.x];
	int query_index = q.query;
	u32* my_noiih = &noiih[query_index];
	u32* my_threshold = &d_threshold[query_index];
	u32* my_passCount = &d_passCount[query_index * num_of_max_count];         //
	u32 my_topk = d_topks[query_index];                //for AT

	T_HASHTABLE* hash_table = &hash_table_list[query_index * hash_table_size];
	u32 * bitmap;
	if (bitmap_bits)
		bitmap = &bitmap_list[query_index * (i_size / (32 / bitmap_bits) + 1)];
	u32 access_id;
	int min, max, order;
	if(q.start_pos >= q.end_pos)
		return;

	min = q.start_pos;
	max = q.end_pos;
	order = q.order;
	bool key_eligible;                //
	bool pass_threshold;    //to determine whether pass the check of my_theshold

	for (int i = 0; i < (max - min - 1) / MATCH_THREADS_PER_BLOCK + 1; ++i)
	{

		int tmp_id = threadIdx.x + i * MATCH_THREADS_PER_BLOCK + min;
		if (tmp_id < max)
		{
			u32 count = 0;                //for AT
			access_id = d_inv[tmp_id];

            if(shift_bits_subsequence != 0)
            {
                int __offset = access_id & (((unsigned int)1<<shift_bits_subsequence) - 1);
                int __new_offset = __offset - order;
                if(__new_offset >= 0)
                {
                    access_id = access_id - __offset + __new_offset;
                }
                else
                    continue;
            }

			u32 thread_threshold = *my_threshold;
            assert(thread_threshold < gridDim.x);

			if (bitmap_bits)
			{

				key_eligible = false;
				//all count are store in the bitmap, and access the count
				count = bitmap_kernel_AT(access_id, bitmap, bitmap_bits,
						thread_threshold, &key_eligible);

				if (!key_eligible)
					continue;                //i.e. count< thread_threshold
			}

			key_eligible = false;
			if (count < *my_threshold)
			{
				continue;      //threshold has been increased, no need to insert
			}

			//Try to find the entry in hash tables
			access_kernel_AT(
					access_id,               
					hash_table, hash_table_size, q, count, &key_eligible,
					my_threshold, &pass_threshold);

			if (key_eligible)
			{
				if (pass_threshold)
				{
					updateThreshold(my_passCount, my_threshold, my_topk, count);
				}

				continue;
			}

			if (!key_eligible)
			{
				//Insert the key into hash table
				//access_id and its location are packed into a packed key

				if (count < *my_threshold)
				{
					continue;//threshold has been increased, no need to insert
				}

				hash_kernel_AT(access_id, hash_table, hash_table_size, q, count,
						my_threshold, my_noiih, overflow, &pass_threshold);
				if (*overflow)
				{

					return;
				}
				if (pass_threshold)
				{
					updateThreshold(my_passCount, my_threshold, my_topk, count);
				}
			}

		}
	}
}
//end for AT

int build_queries(vector<query>& queries, inv_table& table,
		vector<query::dim>& dims, int max_load)
{
    try{
        u64 query_build_start, query_build_stop;
        query_build_start = getTime();

		int max_count = -1;
		for (unsigned int i = 0; i < queries.size(); ++i)
		{
			if (queries[i].ref_table() != &table)
				throw GPUGenie::cpu_runtime_error("Can't build queries. Queries constructed for different table!");
			if (table.build_status() == inv_table::builded)
			{
				if(table.shift_bits_sequence != 0)
				{
					queries[i].build_sequence();// For sequence, balance have not been done
				} else if (queries[i].use_load_balance)
				{
					queries[i].build_and_apply_load_balance(max_load);
				}else
				{
					queries[i].build();
				}
			}
		
			int prev_size = dims.size();
			queries[i].dump(dims);

			int count = dims.size() - prev_size;

			if(count > max_count) max_count = count;
		}

        query_build_stop = getTime();
        Logger::log(Logger::INFO, ">>>>[time profiling]: match: build_queries function takes %f ms. ",
                getInterval(query_build_start, query_build_stop));

        Logger::log(Logger::DEBUG, " dims size: %d.", dims.size());

		return max_count;

	} catch(std::bad_alloc &e){
		throw GPUGenie::cpu_bad_alloc(e.what());
	} catch(GPUGenie::cpu_runtime_error &e){
		throw e;
	} catch(std::exception &e){
		throw GPUGenie::cpu_runtime_error(e.what());
	}
}

int cal_max_topk(vector<query>& queries)
{
	int max_topk = 0;
	for(vector<query>::iterator it = queries.begin(); it != queries.end(); ++it)
	{
		if(it->topk() > max_topk) max_topk = it->topk();
	}
	return max_topk;
}



void match(inv_table& table, vector<query>& queries,
		device_vector<data_t>& d_data, device_vector<u32>& d_bitmap,
		int hash_table_size, int max_load, int bitmap_bits,	//or for AT: for adaptiveThreshold, if bitmap_bits<0, use adaptive threshold, the absolute value of bitmap_bits is count value stored in the bitmap
		device_vector<u32>& d_noiih, device_vector<u32>& d_threshold, device_vector<u32>& d_passCount)
{
	try{
        u32 shift_bits_subsequence = table._shift_bits_subsequence();

        if (table.build_status() == inv_table::not_builded)
            throw GPUGenie::cpu_runtime_error("table not built!");
        
        // Time measuring events
        cudaEvent_t kernel_start, kernel_stop;
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
		u64 match_stop, match_start;
        match_start = getTime();

        Logger::log(Logger::INFO, "[  0%] Starting matching...");

		u32 num_of_max_count=0, max_topk=0;
		u32 loop_count = 1u;
		d_noiih.resize(queries.size(), 0);
		u32 * d_noiih_p = thrust::raw_pointer_cast(d_noiih.data());

		vector<query::dim> dims;

		Logger::log(Logger::DEBUG, "hash table size: %d.", hash_table_size);
		u64 match_query_start, match_query_end;
		match_query_start = getTime();
		num_of_max_count = build_queries(queries, table, dims, max_load);

		match_query_end = getTime();
		Logger::log(Logger::INFO,
				">>>>[time profiling]: match: build_queries function takes %f ms. ",
				getInterval(match_query_start, match_query_end));
		Logger::log(Logger::DEBUG, " dims size: %d.",
				dims.size());

		//for AT: for adaptiveThreshold, enable adaptiveThreshold
		if (bitmap_bits < 0)
		{
			bitmap_bits = -bitmap_bits;
			//for hash_table_size, still let it determine by users currently
		}

		Logger::log(Logger::DEBUG,
				"[info] bitmap_bits:%d.",
				bitmap_bits);

		//end for AT

		int threshold = bitmap_bits - 1, bitmap_size = 0;
		if (bitmap_bits > 1)
		{
			float logresult = std::log2((float) bitmap_bits);
			bitmap_bits = (int) logresult;
			if (logresult - bitmap_bits > 0)
			{
				bitmap_bits += 1;
			}
			logresult = std::log2((float) bitmap_bits);
			bitmap_bits = (int) logresult;
			if (logresult - bitmap_bits > 0)
			{
				bitmap_bits += 1;
			}
			bitmap_bits = pow(2, bitmap_bits);
			bitmap_size = ((((unsigned int)1<<shift_bits_subsequence) * table.i_size()) / (32 / bitmap_bits) + 1)
					* queries.size();
		}
		else
		{
			bitmap_bits = threshold = 0;
		}

		Logger::log(Logger::DEBUG, "Bitmap bits: %d, threshold:%d.", bitmap_bits,
				threshold);
		Logger::log(Logger::INFO, "[ 20%] Declaring device memory...");

		d_bitmap.resize(bitmap_size);


        cout << "query_transfer time = " ; 
        u64 query_start = getTime();

		device_vector<query::dim> d_dims(dims);
		query::dim* d_dims_p = raw_pointer_cast(d_dims.data());

        u64 query_end = getTime();
        cout << getInterval(query_start, query_end) << "ms." << endl;

        u64 dataTransferStart, dataTransferEnd;
        dataTransferStart = getTime();
        if (table.get_total_num_of_table() > 1 || !table.is_stored_in_gpu)
        {
            table.cpy_data_to_gpu();
        }
        dataTransferEnd  = getTime();
        
        if (bitmap_size)
        {
            thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
        }
        u32 * d_bitmap_p = raw_pointer_cast(d_bitmap.data());


		Logger::log(Logger::INFO, "[ 30%] Allocating device memory to tables...");

		data_t nulldata;
		nulldata.id = 0u;
		nulldata.aggregation = 0.0f;
		T_HASHTABLE* d_hash_table;
		data_t* d_data_table;
		d_data.clear();

		d_data.resize(queries.size() * hash_table_size, nulldata);
		d_data_table = thrust::raw_pointer_cast(d_data.data());
		d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);

		Logger::log(Logger::INFO, "[ 33%] Copying memory to symbol...");

		cudaCheckErrors(cudaMemcpyToSymbol(d_offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));

		Logger::log(Logger::INFO,"[ 40%] Starting match kernels...");
		cudaEventRecord(kernel_start);

		bool h_overflow[1] = {false};
		bool * d_overflow;

		cudaCheckErrors(cudaMalloc((void**) &d_overflow, sizeof(bool)));

		do
		{
			h_overflow[0] = false;
			cudaCheckErrors(cudaMemcpy(d_overflow, h_overflow, sizeof(bool), cudaMemcpyHostToDevice));
            d_threshold.resize(queries.size());
            thrust::fill(d_threshold.begin(), d_threshold.end(), 1);
            u32 * d_threshold_p = thrust::raw_pointer_cast(d_threshold.data());

            //which num_of_max_count should be used?

            //num_of_max_count = dims.size();
            
            d_passCount.resize(queries.size()*num_of_max_count);
            thrust::fill(d_passCount.begin(), d_passCount.end(), 0u);
            u32 * d_passCount_p = thrust::raw_pointer_cast(d_passCount.data());
            max_topk = cal_max_topk(queries);
            device_vector<u32> d_topks;
            d_topks.resize(queries.size());
            thrust::fill(d_topks.begin(), d_topks.end(), max_topk);
            u32 * d_topks_p = thrust::raw_pointer_cast(d_topks.data());


            match_AT<<<dims.size(), MATCH_THREADS_PER_BLOCK>>>
            (table.m_size(),
                    table.i_size() * ((unsigned int)1<<shift_bits_subsequence),
                    hash_table_size,
                    table.d_inv_p,
                    d_dims_p,
                    d_hash_table,
                    d_bitmap_p,
                    bitmap_bits,
                    d_topks_p,
                    d_threshold_p,//initialized as 1, and increase gradually
                    d_passCount_p,//initialized as 0, count the number of items passing one d_threshold
                    num_of_max_count,//number of maximum count per query
                    d_noiih_p,
                    d_overflow,
                    shift_bits_subsequence);
            cudaCheckErrors(cudaDeviceSynchronize());
			cudaCheckErrors(cudaMemcpy(h_overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));

			if(h_overflow[0])
			{
				hash_table_size += num_of_max_count*max_topk;
				if(hash_table_size > table.i_size())
				{
					hash_table_size = table.i_size();
				}
				thrust::fill(d_noiih.begin(), d_noiih.end(), 0u);
				if(bitmap_size)
				{
					thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
				}
				d_data.resize(queries.size()*hash_table_size);
				thrust::fill(d_data.begin(), d_data.end(), nulldata);
				d_data_table = thrust::raw_pointer_cast(d_data.data());
				d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);
			}

			if (loop_count>1 || (loop_count == 1 && h_overflow[0]))
			{
				Logger::log(Logger::INFO,"%d time trying to launch match kernel: %s!", loop_count, h_overflow[0]?"failed":"succeeded");
			}
			loop_count ++;

		} while (h_overflow[0]);

        cudaCheckErrors(cudaFree(d_overflow));

		cudaEventRecord(kernel_stop);
		Logger::log(Logger::INFO,"[ 90%] Starting data converting......");

		convert_to_data<<<hash_table_size*queries.size() / 1024 + 1,1024>>>(d_hash_table,(u32)hash_table_size*queries.size());

		Logger::log(Logger::INFO, "[100%] Matching is done!");

		match_stop = getTime();

		cudaEventSynchronize(kernel_stop);
		float kernel_elapsed = 0.0f;
		cudaEventElapsedTime(&kernel_elapsed, kernel_start, kernel_stop);
		Logger::log(Logger::INFO,
				">>>>[time profiling]: Match kernel takes %f ms. (GPU running) ",
				kernel_elapsed);
		Logger::log(Logger::INFO,
				">>>>[time profiling]: Match function takes %f ms.  (including Match kernel, GPU+CPU part)",
				getInterval(match_start, match_stop));
		Logger::log(Logger::VERBOSE, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

        PerfLogger<MatchingPerfData>::Instance().Log()
            .OverallTime(getInterval(match_start, match_stop))
            .QueryCompilationTime(getInterval(match_query_start, match_query_end))
            .QueryTransferTime(getInterval(query_start, query_end))
            .DataTransferTime(getInterval(dataTransferStart, dataTransferEnd))
            .MatchingTime(kernel_elapsed)
            .InvSize(sizeof(int) * table.inv()->size())
            .DimsSize(dims.size() * sizeof(query::dim))
            .HashTableCapacityPerQuery(hash_table_size)
            .ThresholdSize(queries.size() * sizeof(u32))
            .PasscountSize(queries.size() * num_of_max_count * sizeof(u32))
            .BitmapSize(bitmap_size * sizeof(u32))
            .NumItemsInHashTableSize(queries.size() * sizeof(u32))
            .TopksSize(queries.size() * sizeof(u32))
            .HashTableSize(queries.size() * hash_table_size * sizeof(data_t));

	} catch(std::bad_alloc &e){
		throw GPUGenie::gpu_bad_alloc(e.what());
	}
}


// debug use
static int build_queries_direct(vector<GPUGenie::query> &queries, GPUGenie::inv_table &table, vector<GPUGenie::query::dim> &dims)
{
	try
	{
		int max_count = -1;
		for (size_t i = 0; i < queries.size(); ++i)
		{
			if (queries[i].ref_table() != &table)
				throw GPUGenie::cpu_runtime_error("table not built");
			int prev_size = dims.size();
			if (table.build_status() == GPUGenie::inv_table::builded)
				queries[i].build(dims); // overloaded
			int count = dims.size() - prev_size;
			if (count > max_count)
				max_count = count;
		}

		return max_count;
	}
	catch (std::bad_alloc &e)
	{
		throw GPUGenie::cpu_bad_alloc(e.what());
	}
	catch (GPUGenie::cpu_runtime_error &e)
	{
		throw e;
	}
	catch (std::exception &e)
	{
		throw GPUGenie::cpu_runtime_error(e.what());
	}
}

void
match_MT(vector<inv_table*>& table, vector<vector<query> >& queries,
		vector<device_vector<data_t> >& d_data, vector<device_vector<u32> >& d_bitmap,
		vector<int>& hash_table_size, vector<int>& max_load, int bitmap_bits,
		vector<device_vector<u32> >& d_noiih, vector<device_vector<u32> >& d_threshold,
		vector<device_vector<u32> >& d_passCount, size_t start, size_t finish)
{
	try
	{
		/* timing */
		u64 match_stop, match_start;
		cudaEvent_t kernel_start, kernel_stop;
		float kernel_elapsed;
		cudaEventCreate(&kernel_start);
		cudaEventCreate(&kernel_stop);
		match_start = getTime();
		Logger::log(Logger::INFO, "[  0%] Starting matching...");

		/* variable declaration */
		u32 shift_bits_subsequence = table.at(0)->_shift_bits_subsequence();
		vector<vector<query::dim> > dims(table.size()); /* query::dim on CPU */
		vector<device_vector<query::dim> > d_dims(table.size()); /* query::dim on GPU */
		vector<query::dim*> d_dims_p(table.size()); /* query::dim pointers */
		vector<u32*> d_noiih_p(table.size());
		vector<u32> num_of_max_count(table.size(), 0);
		vector<u32> max_topk(table.size(), 0);
		vector<u32*> d_bitmap_p(table.size());
		vector<bool*> d_overflow(table.size());
		vector<T_HASHTABLE*> d_hash_table(table.size());
		vector<u32*> d_threshold_p(table.size());
		vector<u32*> d_passCount_p(table.size());
		vector<device_vector<u32> > d_topks(table.size());
		vector<u32*> d_topks_p(table.size());
		vector<int> threshold(table.size(), bitmap_bits - 1);
		vector<int> bitmap_size(table.size(), 0);
		data_t nulldata = {0u, 0.0f};
		data_t* d_data_table;

		/* adaptive threshold */
		if (bitmap_bits < 0)
			//for hash_table_size, still let it determine by users currently
			bitmap_bits = -bitmap_bits;
		Logger::log(Logger::DEBUG,
				"[info] bitmap_bits:%d.",
				bitmap_bits);
		int bitmap_bits_copy = bitmap_bits;

		/* table dependent variable pre-processing */
		for (size_t i = start; i < finish; ++i)
		{
			if (queries.at(i).empty())
				continue;
			if (table.at(i)->build_status() == inv_table::not_builded)
				throw GPUGenie::cpu_runtime_error("table not built!");

			/* bitmap */
			bitmap_bits = bitmap_bits_copy;
			if (bitmap_bits > 1)
			{
				float logresult = std::log2((float) bitmap_bits);
				bitmap_bits = (int) logresult;
				if (logresult - bitmap_bits > 0)
					bitmap_bits += 1;
				logresult = std::log2((float) bitmap_bits);
				bitmap_bits = (int) logresult;
				if (logresult - bitmap_bits > 0)
					bitmap_bits += 1;
				bitmap_bits = pow(2, bitmap_bits);
				bitmap_size[i] = ((((unsigned int)1<<shift_bits_subsequence) * table.at(i)->i_size()) / (32 / bitmap_bits) + 1)
						* queries.at(i).size();
			}
			else
				bitmap_bits = threshold[i] = 0;

			Logger::log(Logger::DEBUG, "[ 20%] Declaring device memory...");
			d_bitmap.at(i).resize(bitmap_size.at(i));
			d_bitmap_p[i] = thrust::raw_pointer_cast(d_bitmap.at(i).data());

			/* number of items in hashtable */
			d_noiih.at(i).resize(queries.at(i).size(), 0u);
			d_noiih_p[i] = thrust::raw_pointer_cast(d_noiih.at(i).data());
			Logger::log(Logger::DEBUG, "hash table size: %d.", hash_table_size.at(i));

			/* build query */
			u64 match_query_start, match_query_end;
			match_query_start = getTime();
			num_of_max_count[i] = build_queries_direct(queries.at(i), table.at(i)[0], dims.at(i));
			//num_of_max_count[i] = build_q(queries.at(i), table.at(i)[0], dims.at(i), max_load.at(i));
			match_query_end = getTime();
			Logger::log(Logger::DEBUG,
					">>>>[time profiling]: match: build_queries function takes %f ms. ",
					getInterval(match_query_start, match_query_end));
			Logger::log(Logger::DEBUG, " dims size: %d.",
					dims.at(i).size());

			/* transfer query */
			u64 query_start = getTime();
			d_dims[i] = dims.at(i);
			//vector<query::dim>().swap(dims.at(i));
			d_dims_p[i] = raw_pointer_cast(d_dims.at(i).data());
			u64 query_end = getTime();
			//clog << "query_transfer time = " << getInterval(query_start, query_end) << "ms." << endl;

			Logger::log(Logger::DEBUG, "[ 30%] Allocating device memory to tables...");

			/* hashtable */
			d_data.at(i).clear();
			d_data.at(i).resize(queries.at(i).size() * hash_table_size.at(i), nulldata);
			d_data_table = thrust::raw_pointer_cast(d_data.at(i).data());
			d_hash_table[i] = reinterpret_cast<T_HASHTABLE*>(d_data_table);

			/* overflow */
			bool f = false;
			cudaCheckErrors(cudaMalloc((void**)&d_overflow[i], sizeof(bool)));
			cudaCheckErrors(cudaMemcpy(d_overflow[i], &f, sizeof(bool), cudaMemcpyHostToDevice));

			/* threshold */
			d_threshold.at(i).resize(queries.at(i).size());
			thrust::fill(d_threshold.at(i).begin(), d_threshold.at(i).end(), 1);
			d_threshold_p[i] = thrust::raw_pointer_cast(d_threshold.at(i).data());

			/* zipper array */
			d_passCount.at(i).resize(queries.at(i).size() * num_of_max_count.at(i));
			thrust::fill(d_passCount.at(i).begin(), d_passCount.at(i).end(), 0u);
			d_passCount_p[i] = thrust::raw_pointer_cast(d_passCount.at(i).data());

			/* topk */
			max_topk[i] = cal_max_topk(queries.at(i));
			d_topks.at(i).resize(queries.at(i).size());
			thrust::fill(d_topks.at(i).begin(), d_topks.at(i).end(), max_topk[i]);
			d_topks_p[i] = thrust::raw_pointer_cast(d_topks.at(i).data());
		}

		/* offset */
		Logger::log(Logger::INFO, "[ 33%] Copying memory to symbol...");

		cudaCheckErrors(cudaMemcpyToSymbol(d_offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));


		/* match kernel */
		Logger::log(Logger::INFO,"[ 40%] Starting match kernels...");
		cudaEventRecord(kernel_start);
		for (size_t i = start; i < finish; ++i)
		{
			if (queries.at(i).empty())
				continue;
			match_AT<<<dims.at(i).size(), MATCH_THREADS_PER_BLOCK>>>(
				table.at(i)->m_size(),
				table.at(i)->i_size() * ((unsigned int)1<<shift_bits_subsequence),
				hash_table_size.at(i),
				table.at(i)->d_inv_p,
				d_dims_p.at(i),
				d_hash_table.at(i),
				d_bitmap_p.at(i),
				bitmap_bits,
				d_topks_p.at(i),
				d_threshold_p.at(i), //initialized as 1, and increase gradually
				d_passCount_p.at(i), //initialized as 0, count the number of items passing one d_threshold
				num_of_max_count.at(i), //number of maximum count per query
				d_noiih_p.at(i),
				d_overflow.at(i),
				shift_bits_subsequence);
		}
		cudaCheckErrors(cudaDeviceSynchronize());

		/* clean up */
		for (size_t i = start; i < finish; ++i)
			cudaCheckErrors(cudaFree(d_overflow.at(i)));

		cudaEventRecord(kernel_stop);
		Logger::log(Logger::INFO,"[ 90%] Starting data converting......");

		for (size_t i = start; i < finish; ++i)
			convert_to_data<<<hash_table_size.at(i) * queries.at(i).size() / 1024 + 1, 1024>>>(d_hash_table.at(i), (u32)hash_table_size.at(i)*queries.at(i).size());

		Logger::log(Logger::INFO, "[100%] Matching is done!");

		match_stop = getTime();
		cudaEventSynchronize(kernel_stop);

		kernel_elapsed = 0.0f;
		cudaEventElapsedTime(&kernel_elapsed, kernel_start, kernel_stop);
		Logger::log(Logger::INFO,
				">>>>[time profiling]: Match kernel takes %f ms. (GPU running) ",
				kernel_elapsed);
		Logger::log(Logger::INFO,
				">>>>[time profiling]: Match function takes %f ms.  (including Match kernel, GPU+CPU part)",
				getInterval(match_start, match_stop));
		Logger::log(Logger::VERBOSE, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
	} catch(std::bad_alloc &e){
		throw GPUGenie::gpu_bad_alloc(e.what());
	}
}

}
