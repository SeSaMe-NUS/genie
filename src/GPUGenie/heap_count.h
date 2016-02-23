#ifndef _HEAP_COUNT_H
#define _HEAP_COUNT_H

#include <thrust/device_vector.h>
#include "match.h"

void heap_count_topk(thrust::device_vector<data_t>& d_data,
					 thrust::device_vector<data_t>& d_topk,
					 thrust::device_vector<u32>& d_threshold,
					 thrust::device_vector<u32>& d_passCount,
					 int topk,
					 int num_of_queries);

#endif
