#ifndef UPDATE_MIN_MAX_INC
#define UPDATE_MIN_MAX_INC

static __global__ void update_min_max(float* min, float* max, const float MIN, const float MAX, int* last_bucket_index, const int number_of_buckets);
static __global__ void update_min_max(float* min, float* max, int* last_bucket_index, const int number_of_buckets);

static __global__ void update_min_max(float* min, float* max, const float MIN, const float MAX, int* last_bucket_index, const int number_of_buckets)
{
	int bid = blockIdx.x;
	float avg = (MAX - MIN)/number_of_buckets;
	min[bid] = MIN + avg*(last_bucket_index[bid]%number_of_buckets);
	max[bid] = min[bid] + avg;
}

static __global__ void update_min_max(float* min, float* max, int* last_bucket_index, const int number_of_buckets)
{
	int bid = blockIdx.x;
	float avg = (max[bid] - min[bid])/number_of_buckets;
	min[bid] = min[bid] + avg*(last_bucket_index[bid]%number_of_buckets);
	max[bid] = min[bid] + avg;
}
#endif
