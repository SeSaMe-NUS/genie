#ifndef ASSIGN_BUCKET_
#define ASSIGN_BUCKET_

template<typename Data, typename GetValue>
static __global__ void assign_bucket(Data* data, GetValue sel, Bucket* bucket, int* start_index, int* end_index, int* bucket_size, float* min, float* max, int number_of_buckets);
template<typename Data, typename GetValue>
static __global__ void reassign_bucket(Data* data, GetValue sel, int* data_index, Bucket* bucket, int* end_index, int* bucket_size, float* min, float* max, int number_of_buckets);

template<typename Data, typename GetValue>
static __global__ void assign_bucket(Data* data, GetValue sel, Bucket* bucket, int* start_index, int* end_index, int* bucket_size, float* min, float* max, int number_of_buckets)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int blk_start_index = start_index[bid];
	int blk_bucket_start_index = bid * number_of_buckets;
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
    int offset = blockDim.x;
    int index, bucket_index, bucket_shift_index;
    for(int i=0; i<round; i++)
    {
        index = (blk_start_index+tid)+i*offset;
        bucket_shift_index = index-start_index[0];
        if(index < blk_end_index)
        {
        	float value = sel.valueOf(data[index]);
        	if(value > max[bid]) value = max[bid];
        	if(value < min[bid]) value = min[bid];
            bucket_index = (value - min[bid]) * number_of_buckets / (max[bid]-min[bid]);
            if(bucket_index == number_of_buckets)
            {
                bucket_index = number_of_buckets - 1;
            }
            bucket_index = bucket_index + blk_bucket_start_index;
            bucket[bucket_shift_index].a_index = index;
            bucket[bucket_shift_index].b_index = bucket_index;
            atomicAdd(&bucket_size[bucket_index], 1);
        }
    }
}

template<typename Data, typename GetValue>
static __global__ void reassign_bucket(Data* data, GetValue sel, int* data_index, Bucket* bucket, int* end_index, int* bucket_size, float* min, float* max, int number_of_buckets)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_bucket_start_index = bid * number_of_buckets;
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
    int offset = blockDim.x;
    int index, bucket_index;
    for(int i=0; i<round; i++)
    {
        index = (blk_start_index+tid)+i*offset;
        if(index < blk_end_index)
        {
        	float value = sel.valueOf(data[data_index[index]]);
        	if(value > max[bid]) value = max[bid];
        	if(value < min[bid]) value = min[bid];
            bucket_index = (value - min[bid]) * number_of_buckets / (max[bid]-min[bid]);
            if(bucket_index == number_of_buckets)
            {
                bucket_index = number_of_buckets - 1;
            }
            bucket_index = bucket_index + blk_bucket_start_index;
            bucket[index].a_index = data_index[index];
            bucket[index].b_index = bucket_index;
            atomicAdd(&bucket_size[bucket_index], 1);
        }
    }
}
#endif
