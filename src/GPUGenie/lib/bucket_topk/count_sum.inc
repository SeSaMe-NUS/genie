#ifndef COUNT_SUM_INC
#define COUNT_SUM_INC

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
using namespace thrust;

static void count_sum(device_vector<int> *k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part);
static __global__ void copy_size(int* last_bucket_index, int* bucket_size, int number_of_buckets, int* new_partition_size);
static __global__ void check_index(int* k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss);

static void count_sum(device_vector<int> *k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part)
{
	device_vector<int> d_bucket_size_sum(number_of_buckets*number_of_parts);
	inclusive_scan((*d_bucket_size).begin(), (*d_bucket_size).end(), d_bucket_size_sum.begin());
	check_index<<<number_of_parts, THREADS_PER_BLOCK>>>(
			raw_pointer_cast((*k).data()),
			raw_pointer_cast(d_bucket_size_sum.data()),
			number_of_buckets,
			raw_pointer_cast((*last_bucket_index).data()),
			raw_pointer_cast((*miss).data()));

	copy_size<<<number_of_parts, 1>>>(raw_pointer_cast((*last_bucket_index).data()), raw_pointer_cast((*d_bucket_size).data()), number_of_buckets, raw_pointer_cast((*end_index_of_each_part).data()));
	inclusive_scan((*end_index_of_each_part).begin(), (*end_index_of_each_part).end(), (*end_index_of_each_part).begin());
}

static __global__ void check_index(int* k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int blk_start_index = bid*number_of_buckets;
	int blk_end_index = (bid+1)*number_of_buckets;
	int blk_before_end_index = blk_start_index-1;
	int size_before = blk_before_end_index<0 ? 0 : bucket_size_sum[blk_before_end_index];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(tid==0 && i==0)
			{
				if(bucket_size_sum[index] - size_before > k[bid])
				{
					last_bucket_index[bid] = index;
					miss[bid] = k[bid];
				}
			}
			else if(bucket_size_sum[index] - size_before > k[bid] && bucket_size_sum[index-1] - size_before <= k[bid])
			{
					last_bucket_index[bid] = index;
					miss[bid] = k[bid] - (bucket_size_sum[index-1]- size_before);
			}
			else if(index == blk_end_index-1 && bucket_size_sum[index] - size_before <= k[bid])
			{
				last_bucket_index[bid] = blk_end_index;
				miss[bid] = 0;
			}
		}
	}
}

static __global__ void copy_size(int* last_bucket_index, int* bucket_size, int number_of_buckets, int* new_partition_size)
{
	int bid = blockIdx.x;
	int last = (bid+1)*number_of_buckets;
	if(last_bucket_index[bid] < last)
		new_partition_size[bid] = bucket_size[last_bucket_index[bid]];
	else
		new_partition_size[bid] = 0;
}
#endif
