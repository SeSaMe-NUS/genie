/*! \file heap_count.cu
 *  \brief Implementation for heap_count.h
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <genie/utility/Timing.h>
#include <genie/matching/match.h>
#include <genie/utility/cuda_macros.h>
#include <genie/matching/heap_count.h>

#define THREADS_PER_BLOCK 256
#define GPUGenie_Minus_One_THREADS_PER_BLOCK 1024

using namespace std;
using namespace thrust;
using namespace genie::matching;

__global__
void count_over_threshold(data_t * data, int * result, u32 * thresholds,
		int data_size)
{
	int tid = threadIdx.x;
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	data_t * my_data = data + data_size * blockIdx.x;
	u32 my_threshold = thresholds[blockIdx.x];
	for (int i = 0; i * blockDim.x + tid < data_size; ++i)
	{
		int id = i * blockDim.x + tid;
		if (my_data[id].aggregation > my_threshold)
		{
			result[index]++;
		}
	}
}

__global__
void exclusive_scan(int * data, int * buffer, int * output, int size)
{
	int tid = threadIdx.x;
	int * my_buffer = buffer + 2 * blockIdx.x * size;
	int * my_data = data + blockIdx.x * size;
	int * my_output = output + blockIdx.x * (size + 1);
	my_buffer[size + tid] = my_buffer[tid] = tid == 0 ? 0 : my_data[tid - 1];

	__syncthreads();

	for (int offset = 1; offset < size; offset *= 2)
	{
		if (tid >= offset)
			my_buffer[tid] += my_buffer[size + tid - offset];
		__syncthreads();
		my_buffer[size + tid] = my_buffer[tid];
		__syncthreads();
	}
	my_output[tid] = my_buffer[tid]; // write output
	if (tid == size - 1)
	{
		my_output[tid + 1] = my_output[tid] + my_data[tid];
	}

}

__global__
void fill_in_scan(data_t * data, u32 * thresholds, int * indices, data_t * topk,
		int data_size, int topk_size)
{
	int tid = threadIdx.x;
	data_t * my_data = data + data_size * blockIdx.x;
	u32 my_threshold = thresholds[blockIdx.x];
	data_t * my_topk = topk + blockIdx.x * topk_size
			+ indices[blockIdx.x * (blockDim.x + 1) + tid];
	for (int i = 0; i * blockDim.x + tid < data_size; ++i)
	{
		int id = i * blockDim.x + tid;
		if (my_data[id].aggregation > my_threshold)
		{
			*my_topk = my_data[id];
			my_topk++;
		}
	}
	__syncthreads();

	__shared__ int index[1];
	my_topk = topk + blockIdx.x * topk_size;
	if (tid == 0)
		*index = indices[blockIdx.x * (blockDim.x + 1) + blockDim.x];
	__syncthreads();

	if (*index >= topk_size)
		return;
	for (int i = 0; i * blockDim.x + tid < data_size; ++i)
	{
		if (*index >= topk_size)
			return;
		int id = i * blockDim.x + tid;
		if (int(my_data[id].aggregation) == my_threshold)
		{
			int old_index = atomicAdd(index, 1);
			if (old_index >= topk_size)
			{
				return;
			}
			else
			{
				my_topk[old_index] = my_data[id];
			}
		}
	}
}

__global__
void transform_threshold(u32 * thresholds, int size, int max_count)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= size)
		return;
	if(thresholds[tId] > max_count){
		thresholds[tId] = max_count - 1;
	} else if (thresholds[tId] != 0)
	{
		thresholds[tId]--;
	}
}

void write_hashtable_to_file(thrust::device_vector<data_t>& d_data, int num_of_queries){

	int part_size = d_data.size() / num_of_queries;
	thrust::host_vector<data_t> h_data(d_data);
	FILE * fout = NULL;
	FILE * fout_compact = NULL;
	std::string s = genie::utility::currentDateTime();
	char fout_name[100];
	char fout_compact_name[100];
	sprintf(fout_name, "%s.txt", s.c_str());
	sprintf(fout_compact_name, "%s-compact.txt", s.c_str());
	fout = fopen(fout_name, "w");
	fout_compact = fopen(fout_compact_name, "w");
	for(int qi = 0; qi < num_of_queries; ++qi){
		fprintf(fout, "Query %d:\n", qi);
		fprintf(fout_compact, "Query %d:\n", qi);
		for(int di = 0; di < part_size; ++di){
			int id = qi * part_size + di;
			if(h_data[id].aggregation != 0.0f || h_data[id].id != 0){
				fprintf(fout_compact, "[%d] %d\n", h_data[id].id, int(h_data[id].aggregation));
			}
			fprintf(fout, "[%d] %d\n", h_data[id].id, int(h_data[id].aggregation));
		}
		fprintf(fout, "\n");
		fprintf(fout_compact, "\n");
	}
	fclose(fout);
	fclose(fout_compact);
}

void genie::matching::heap_count_topk(thrust::device_vector<data_t>& d_data,
		thrust::device_vector<data_t>& d_topk,
		thrust::device_vector<u32>& d_threshold,
		thrust::device_vector<u32>& d_passCount, int topk, int num_of_queries)
{
	int data_size = d_data.size() / num_of_queries;
	int threads =
			data_size >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : data_size;
	int max_count = d_passCount.size() / num_of_queries;
	int * d_num_over_threshold_p;
	u32 * d_threshold_p;
	data_t * d_data_p;

	cudaCheckErrors(
			cudaMalloc((void** ) &d_num_over_threshold_p,
					sizeof(int) * threads * num_of_queries));
	cudaCheckErrors(
			cudaMemset((void* ) d_num_over_threshold_p, 0,
					sizeof(int) * threads * num_of_queries));

	//DEBUG

	//write_hashtable_to_file(d_data, num_of_queries);

//	thrust::host_vector<u32> h_threshold_b(d_threshold);
//	printf("Thresholds before minus one transforms:\n");
//	for(int i = 0; i < h_threshold_b.size(); ++i){
//		printf("%d ", h_threshold_b[i]);
//	}
//	printf("\n");
	//End DEBUG

	d_threshold_p = thrust::raw_pointer_cast(d_threshold.data());
	transform_threshold<<<d_threshold.size() / GPUGenie_Minus_One_THREADS_PER_BLOCK + 1,
			GPUGenie_Minus_One_THREADS_PER_BLOCK>>>(d_threshold_p,
			d_threshold.size(), max_count);
	//DEBUG
//	thrust::host_vector<u32> h_threshold_af(d_threshold);
//	printf("Thresholds after minus one transforms:\n");
//	for(int i = 0; i < h_threshold_af.size(); ++i){
//		printf("%d ", h_threshold_af[i]);
//	}
//	printf("\n");
	//End DEBUG

	d_data_p = thrust::raw_pointer_cast(d_data.data());

	count_over_threshold<<<num_of_queries, threads>>>(d_data_p,
			d_num_over_threshold_p, d_threshold_p, data_size);

	//Debugging
//	int *h_result;
//	h_result = (int*) malloc(sizeof(int) * threads * num_of_queries);
//	cudaMemcpy((void*) h_result, d_num_over_threshold_p,
//			sizeof(int) * threads * num_of_queries, cudaMemcpyDeviceToHost);

//	for (int i = 0; i < num_of_queries; ++i)
//	{
//		for (int j = 0; j < threads && j < 15; ++j)
//		{
//			printf("%d ", h_result[i * threads + j]);
//		}
//		printf("\n");
//	}
//	printf("-----------------------------------\n");

	int * d_buffer, *d_scan_indices;
	cudaCheckErrors(cudaMalloc((void**) &d_buffer, 2 * sizeof(int) * threads * num_of_queries));
	cudaCheckErrors(cudaMemset((void*) d_buffer, 0, 2 * sizeof(int) * threads * num_of_queries));
	cudaCheckErrors(cudaMalloc((void**) &d_scan_indices,
			sizeof(int) * (threads + 1) * num_of_queries));

	exclusive_scan<<<num_of_queries, threads>>>(d_num_over_threshold_p,
			d_buffer, d_scan_indices, threads);

	//Debugging
//	free(h_result);
//	h_result = (int*) malloc(sizeof(int) * (threads + 1) * num_of_queries);
//	cudaMemcpy((void*) h_result, d_scan_indices,
//			sizeof(int) * (threads + 1) * num_of_queries,
//			cudaMemcpyDeviceToHost);

//	for (int i = 0; i < num_of_queries; ++i)
//	{
//		for (int j = 0; j < threads + 1 && j < 16; ++j)
//		{
//			printf("%d ", h_result[i * (threads + 1) + j]);
//		}
//		printf("\n");
//	}
//	printf("-----------------------------------\n");

	d_topk.resize(topk * num_of_queries);
	data_t empty_data ={ 0, 0 };
	thrust::fill(d_topk.begin(), d_topk.end(), empty_data);
	data_t * d_topk_p = thrust::raw_pointer_cast(d_topk.data());

	fill_in_scan<<<num_of_queries, threads, 2 * sizeof(int)>>>(d_data_p,
			d_threshold_p, d_scan_indices, d_topk_p, data_size, topk);

//	data_t * h_topk_result;
//	h_topk_result = (data_t*) malloc(sizeof(data_t) * topk * num_of_queries);
//	cudaMemcpy((void*) h_topk_result, d_topk_p,
//			sizeof(data_t) * topk * num_of_queries, cudaMemcpyDeviceToHost);

//	for (int i = 0; i < num_of_queries; ++i)
//	{
//		for (int j = 0; j < topk && j < 10; ++j)
//		{
//			printf("%d ", int(h_topk_result[i * topk + j].aggregation));
//		}
//		printf("\n");
//	}
//	printf("-----------------------------------\n");
//	printf("Number of threads per block launched: %d.\n", threads);

	cudaCheckErrors(cudaFree(d_num_over_threshold_p));
	cudaCheckErrors(cudaFree(d_buffer));
	cudaCheckErrors(cudaFree(d_scan_indices));
	//free(h_result);
	//free(h_topk_result);
}

