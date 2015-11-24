#ifndef SAVE_RESULT_INC_
#define SAVE_RESULT_INC_

#include <limits.h>

static __global__ void save_result(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = start_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index, bucket_shift_index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		bucket_shift_index = index-start_index[0];
		if(index < blk_end_index)
		{
			if(bucket[bucket_shift_index].b_index < last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[bucket_shift_index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_result(Bucket* bucket, int* end_index, int* last_bucket_index, float* min, float* max, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index < last_bucket_index[bid] && position < thread_end_index[threadId] && !BUCKET_TOPK_EQUAL(min[bid], max[bid]))
			{
				result[position] = bucket[index].a_index;
				position++;
			}
			if(bucket[index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId] && BUCKET_TOPK_EQUAL(min[bid], max[bid]))
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_miss_array(Bucket* bucket, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_miss_array(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = start_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index, bucket_shift_index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		bucket_shift_index = index-start_index[0];
		if(index < blk_end_index)
		{
			if(bucket[bucket_shift_index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[bucket_shift_index].a_index;
				position++;
			}
		}
	}
}

static __global__ void update_result_last_position(int* result_last_position, int* thread_end_index)
{
	int bid = blockIdx.x;
	result_last_position[bid] = thread_end_index[(bid+1)*blockDim.x-1];
}

template<typename Data>
static __global__ void shift_result(Data* data, int* start_index, int* result, int* end_index_of_result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int start = start_index[bid];
	int result_start = (bid==0) ? 0 : end_index_of_result[bid-1];
	int result_end = end_index_of_result[bid];

	Data newDatak, oldDatak;

	if(tid+result_start < result_end)
	{
		newDatak = data[result[tid+result_start]];
		oldDatak = data[tid+start];
	}
	__syncthreads();
	if(tid+result_start < result_end)
	{
		data[result[tid+result_start]] = oldDatak;
		data[tid+start] = newDatak;
	}
}

struct Bucket_Topk_ValueOfInt {
	__device__ float valueOf(int data)
	{
		return data==-1 ? INT_MAX : data;
	}
};
template<typename Data, typename GetValue>
static __global__ void shift_result_for(Data* data, int* start_index, int* result, GetValue bucket_topk_valofInt, int* end_index_of_result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int start = start_index[bid];
	int result_start = (bid==0) ? 0 : end_index_of_result[bid-1];
	int result_end = end_index_of_result[bid];
	extern __shared__ int result_tmp[];
	blk_sort_inSharedMemory(result+result_start, result_tmp, bucket_topk_valofInt, result_end-result_start);
	if(tid==0)
	{
		Data t;
		for(int i=result_start; i<result_end; i++)
		{
			if(result[i]!=-1)
			{
				t = data[i-result_start+start];
				data[i-result_start+start] = data[result[i]];
				data[result[i]] = t;
			}
		}
	}
}

#endif
