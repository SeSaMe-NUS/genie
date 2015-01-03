#ifndef TID_PARTITION_
#define TID_PARTITION_

#include <stdio.h>

static __global__ void tid_partition(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* result_thread_partition_size);
static __global__ void tid_partition(Bucket* bucket, int* end_index, int* last_bucket_index, int* miss, float* min, float* max, int* result_thread_partition_size);
static __global__ void miss_partition(Bucket* bucket, int* end_index, int* last_bucket_index, int* miss_thread_partition_size);
static __global__ void miss_partition(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* miss_thread_partition_size);
static __global__ void shift_add(int* start, int* size, int* end, int* last_result_index);
static __global__ void shift_add_miss(int* start, int* size, int* end, int* last_result_index);

static void result_thread_partition(device_vector<Bucket> *bucket,
		device_vector<int> *start_index,
		device_vector<int> *end_index,
		device_vector<int> *last_bucket_index,
		device_vector<int> *last_result_index,
		int number_of_parts,
		device_vector<int> *thread_start_index,
		device_vector<int> *thread_end_index)
{
	device_vector<int> partition_size(THREADS_PER_BLOCK*number_of_parts);

	tid_partition<<<number_of_parts, THREADS_PER_BLOCK>>>(
			rpc(*bucket),
			rpc(*start_index),
			rpc(*end_index),
			rpc(*last_bucket_index),
			rpc(partition_size));

	exclusive_scan(partition_size.begin(), partition_size.end(), (*thread_start_index).begin());
	shift_add<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(*thread_start_index), rpc(partition_size), rpc(*thread_end_index), rpc(*last_result_index));
}

static void result_thread_partition(device_vector<Bucket> *bucket,
		device_vector<int> *end_index,
		device_vector<int> *last_bucket_index,
		device_vector<int> *miss,
		device_vector<float> *min,
		device_vector<float> *max,
		device_vector<int> *last_result_index,
		int number_of_parts,
		device_vector<int> *thread_start_index,
		device_vector<int> *thread_end_index)
{
	device_vector<int> partition_size(THREADS_PER_BLOCK*number_of_parts);

	tid_partition<<<number_of_parts, THREADS_PER_BLOCK>>>(
			raw_pointer_cast((*bucket).data()),
			raw_pointer_cast((*end_index).data()),
			raw_pointer_cast((*last_bucket_index).data()),
			rpc(*miss),
			rpc(*min),
			rpc(*max),
			raw_pointer_cast(partition_size.data()));

	exclusive_scan(partition_size.begin(), partition_size.end(), (*thread_start_index).begin());
	shift_add<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(*thread_start_index), rpc(partition_size), rpc(*thread_end_index), rpc(*last_result_index));
}

static void miss_thread_parition(device_vector<Bucket> *bucket,
		device_vector<int> *end_index,
		device_vector<int> *last_bucket_index,
		device_vector<int> *end_index_of_each_part,
		int number_of_parts,
		device_vector<int> *thread_start_index,
		device_vector<int> *thread_end_index)
{
	device_vector<int> partition_size(THREADS_PER_BLOCK*number_of_parts);

	miss_partition<<<number_of_parts, THREADS_PER_BLOCK>>>(
				rpc(*bucket),
				rpc(*end_index),
				rpc(*last_bucket_index),
				rpc(partition_size));
	exclusive_scan(partition_size.begin(), partition_size.end(), (*thread_start_index).begin());
	shift_add_miss<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(*thread_start_index), rpc(partition_size), rpc(*thread_end_index), rpc(*end_index_of_each_part));
}

static void miss_thread_parition(device_vector<Bucket> *bucket,
		device_vector<int> *start_index,
		device_vector<int> *end_index,
		device_vector<int> *last_bucket_index,
		device_vector<int> *end_index_of_each_part,
		int number_of_parts,
		device_vector<int> *thread_start_index,
		device_vector<int> *thread_end_index)
{
	device_vector<int> partition_size(THREADS_PER_BLOCK*number_of_parts);

	miss_partition<<<number_of_parts, THREADS_PER_BLOCK>>>(
				rpc(*bucket),
				rpc(*start_index),
				rpc(*end_index),
				rpc(*last_bucket_index),
				rpc(partition_size));
	exclusive_scan(partition_size.begin(), partition_size.end(), (*thread_start_index).begin());
	shift_add_miss<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(*thread_start_index), rpc(partition_size), rpc(*thread_end_index), rpc(*end_index_of_each_part));
}


static __global__ void tid_partition(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* result_thread_partition_size)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = start_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	result_thread_partition_size[threadId] = 0;
	int index, bucket_shift_index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		bucket_shift_index = index - start_index[0];
		if(index < blk_end_index)
		{
			if(bucket[bucket_shift_index].b_index < last_bucket_index[bid])
			{
				result_thread_partition_size[threadId]++;
			}
		}
	}
}

static __global__ void tid_partition(Bucket* bucket, int* end_index, int* last_bucket_index, int* miss, float* min, float* max, int* result_thread_partition_size)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	result_thread_partition_size[threadId] = 0;
	if(BUCKET_TOPK_EQUAL(min[bid], max[bid]))
	{
		if(miss[bid] < blockDim.x)
		{
			if(tid < miss[bid])
				result_thread_partition_size[threadId] = 1;
			else
				result_thread_partition_size[threadId] = 0;
		}
		else
		{
			if(tid != blockDim.x - 1)
			{
				result_thread_partition_size[threadId] = (blockDim.x-(miss[bid]%blockDim.x) + miss[bid]) / blockDim.x ;
			}
			else
			{
				result_thread_partition_size[threadId] = miss[bid] % blockDim.x;
			}
		}
		__syncthreads();
		miss[bid] = 0;
	}
	else
	{
		int index;
		for(int i=0; i<round; i++)
		{
			index = (blk_start_index+tid)+i*offset;
			if(index < blk_end_index)
			{
				if(bucket[index].b_index < last_bucket_index[bid])
				{
					result_thread_partition_size[threadId]++;
				}
			}
		}
	}
}

static __global__ void miss_partition(Bucket* bucket, int* end_index, int* last_bucket_index, int* miss_thread_partition_size)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	miss_thread_partition_size[threadId] = 0;
	int index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index == last_bucket_index[bid])
			{
				miss_thread_partition_size[threadId]++;
			}
		}
	}
}

static __global__ void miss_partition(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* miss_thread_partition_size)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = start_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	miss_thread_partition_size[threadId] = 0;
	int index, bucket_shift_index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		bucket_shift_index = index - start_index[0];
		if(index < blk_end_index)
		{
			if(bucket[bucket_shift_index].b_index == last_bucket_index[bid])
			{
				miss_thread_partition_size[threadId]++;
			}
		}
	}
}

static __global__ void shift_add(int* start, int* size, int* end, int* last_result_index)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = bid*blockDim.x;
	int size_before = bid==0 ? 0 : start[blk_start_index];
	__syncthreads();
	start[threadId] = start[threadId] - size_before + ((last_result_index==NULL) ? 0 : last_result_index[bid]);
	end[threadId] = start[threadId] + size[threadId];
}

static __global__ void shift_add_miss(int* start, int* size, int* end, int* last_result_index)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid*blockDim.x + tid;

	int blk_start_index = bid*blockDim.x;
	int size_before = bid==0 ? 0 : start[blk_start_index];
	__syncthreads();
	start[threadId] = start[threadId] - size_before + ((bid==0) ? 0 : last_result_index[bid-1]);
	end[threadId] = start[threadId] + size[threadId];
}
#endif
