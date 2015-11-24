#include "knn.h"
#include <math.h>
#include <assert.h>
#include <thrust/copy.h>

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
void
convert_data(float * dd, data_t * od, int size)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if(tId >= size) return;
	dd[tId] = od[tId].aggregation;
}

__global__
void
extract_index(int * id, data_t * od, int size)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if(tId >= size) return;
	//if(od[id[tId]].aggregation != 0.0f)
		id[tId] = od[id[tId]].id;
	//else id[tId] = -1;
}

__global__
void
augment_bitmap(u32 * augmented,
			   u32 * selected,
			   u32 selected_size,
			   u32 augmented_size,
			   u32 num_per_u32)
{
	u32 id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id >= selected_size) return;
	u32 begin = id * GPUGenie_knn_DEFAULT_DATA_PER_THREAD;
	u32 data;
	u32 aug_id;
	u32 offset;
	for(u32 i = begin; i < begin+GPUGenie_knn_DEFAULT_DATA_PER_THREAD; ++i)
	{
		data = selected[i];
		aug_id = i * num_per_u32;
		if(aug_id >= augmented_size) return;
		for(u32 j = aug_id; j < aug_id + num_per_u32 && j < augmented_size; ++j)
		{
			offset = (32u/num_per_u32) * (j - aug_id);
			augmented[j] = (data >> offset) & ((1u << (32u/num_per_u32)) - 1u);
		}
	}
}

__global__
void
correct_index(int * index, int size, int k, int offset)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id >= size) return;

	index[id] -= offset * (id / k);
}
int
GPUGenie::calculate_bits_per_data(int bitmap_bits)
{
	  float logresult = log2((float) bitmap_bits);
	  bitmap_bits = (int) logresult;
	  if(logresult - bitmap_bits > 0)
	  {
		 bitmap_bits += 1;
	  }
	  logresult = log2((float)bitmap_bits);
	  bitmap_bits = (int) logresult;
	  if(logresult - bitmap_bits > 0)
	  {
		 bitmap_bits += 1;
	  }
	  bitmap_bits = pow(2, bitmap_bits);
	  return bitmap_bits;
}

void
GPUGenie::knn(GPUGenie::inv_table& table,
		   vector<GPUGenie::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int max_load)
{
	int hash_table_size = GPUGenie_knn_DEFAULT_HASH_TABLE_SIZE * table.i_size() + 1;
	knn(table, queries, d_top_indexes, hash_table_size, max_load,GPUGenie_knn_DEFAULT_BITMAP_BITS);
}

void
GPUGenie::knn(GPUGenie::inv_table& table, vector<GPUGenie::query>& queries,
    device_vector<int>& d_top_indexes, int hash_table_size, int max_load,int bitmap_bits)
{
	knn(table, queries, d_top_indexes, hash_table_size,max_load, bitmap_bits, table.m_size(), 0, 0);
}


void
GPUGenie::knn(GPUGenie::inv_table& table,
		   vector<GPUGenie::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
		   int max_load,
		   int bitmap_bits,
		   int dim)
{
	knn(table, queries, d_top_indexes, hash_table_size, max_load,bitmap_bits, dim, 0,0);
}
void
GPUGenie::knn_tweets(GPUGenie::inv_table& table,
		   vector<GPUGenie::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
		   int max_load,
		   int bitmap_bits,
		   int dim,
		   int num_of_hot_dims,
		   int hot_dim_threshold)
{
  int qmax = 0;

  for(int i = 0; i < queries.size(); ++i)
  {
	 int count = queries[i].count_ranges();
	  if(count > qmax)
		  qmax = count;
  }
#ifdef GPUGENIE_DEBUG
  u64 start = getTime();
#endif
  knn(table, queries, d_top_indexes, hash_table_size,max_load, bitmap_bits,
		  	  float(qmax+1), num_of_hot_dims, hot_dim_threshold);
#ifdef GPUGENIE_DEBUG
  u64 end = getTime();
  double elapsed = getInterval(start, end);
  printf(">>>>>>> knn takes %fms <<<<<< \n", elapsed);
#endif
}
void
GPUGenie::knn(GPUGenie::inv_table& table,
		   vector<GPUGenie::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
		   int max_load,
		   int bitmap_bits,
		   int dim,
		   int num_of_hot_dims,
		   int hot_dim_threshold)
{
#ifdef GPUGENIE_DEBUG
  printf("Parameters: %d,%d,%d,%d,%d\n", hash_table_size, bitmap_bits, dim, num_of_hot_dims, hot_dim_threshold);
#endif
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
#ifdef GPUGENIE_DEBUG  //for improve
  u64 startKnn = getTime();
#endif

#ifdef GPUGENIE_DEBUG  //for improve
  u64 end3Knn = getTime();
  printf(">>>>> knn() before match() %f ms <<<<<\n", getInterval(startKnn, end3Knn));
#endif

  int bitmap_threshold = bitmap_bits;
  device_vector<data_t> d_data;
  device_vector<u32> d_bitmap;
  //device_vector<u32> d_selected_bitmap;//for improve
  //device_vector<u32> d_augmented_bitmap;//for improve

 // std::vector<int> selected_query_index;//for improve
  //device_vector<int> d_selected_top_indexes;//for improve
#ifdef GPUGENIE_DEBUG  //for improve
  u64 end2Knn = getTime();
  printf(">>>>> knn() before match() %f ms <<<<<\n", getInterval(startKnn, end2Knn));
#endif
  device_vector<u32> d_num_of_items_in_hashtable(queries.size());
  printf("[knn] max_load is %d.\n", max_load);
  match(table, queries, d_data, d_bitmap, hash_table_size,max_load, bitmap_bits, num_of_hot_dims, hot_dim_threshold, d_num_of_items_in_hashtable);
#ifdef GPUGENIE_DEBUG  //for improve
  u64 end1Knn = getTime();
  printf(">>>>> knn() after match() %f ms <<<<<\n", getInterval(startKnn, end1Knn));
#endif
  //cudaCheckErrors(cudaDeviceSynchronize());

  //host_vector<u32> h_num_of_items_in_hashtable(d_num_of_items_in_hashtable);//for improve
  /**   Debug Section  **/
  // printf("The item count in hash map is:\n");
  // for(int i = 0; i < h_num_of_items_in_hashtable.size(); ++i)
  // {
	 //  printf("%d, ", h_num_of_items_in_hashtable[i]);
  // }
  // printf("\n");
  /** End of Debug Section **/

  //If no bitmap, then no need to collect topk in bitmap
  if(bitmap_bits > 1){
	  device_vector<u32> d_augmented_bitmap;//for improve
	  device_vector<u32> d_selected_bitmap;//for improve
	  std::vector<int> selected_query_index;//for improve
	  device_vector<int> d_selected_top_indexes;//for improve
	  host_vector<u32> h_num_of_items_in_hashtable(d_num_of_items_in_hashtable);//for improve
#ifdef GPUGENIE_DEBUG
	  u64 start = getTime();
#endif

	  //Calculate how many bits a data point is assigned to
	  bitmap_bits = calculate_bits_per_data(bitmap_bits);
	  u32 bitmap_size = table.i_size() / (32/bitmap_bits) + 1;
	  u32 num_per_u32 = 32u / bitmap_bits;

	  assert(32u % bitmap_bits == 0);

	  //Collect bitmap of queries with no enough topk result
	  for(u32 i = 0; i < h_num_of_items_in_hashtable.size(); ++i)
	  {
		  if(h_num_of_items_in_hashtable[i] < queries[i].topk())
		  {
			  selected_query_index.push_back(i);
			  d_selected_bitmap.insert(d_selected_bitmap.end(),
									   d_bitmap.begin()+i*bitmap_size,
									   d_bitmap.begin()+(i+1)*bitmap_size);
		  }
	  }
	  d_bitmap.clear();
	  device_vector<u32>().swap(d_bitmap);

#ifdef GPUGENIE_DEBUG
	  u64 end = getTime();
	  printf("size of selected bitmap: %d\n", d_selected_bitmap.size());
	  printf(">>>>> extract selected bitmaps takes %fms <<<<<\n", getInterval(start, end));
#endif

	  if(selected_query_index.size() > 0)
	  {
#ifdef GPUGENIE_DEBUG
		  start = getTime();
#endif
		  //Augment the selected bitmap
		  u32 augmented_size = selected_query_index.size() * table.i_size();
		  d_augmented_bitmap.resize(augmented_size);
		  u32 * d_selected_bitmap_p = thrust::raw_pointer_cast(d_selected_bitmap.data());
		  u32 * d_augmented_bitmap_p = thrust::raw_pointer_cast(d_augmented_bitmap.data());
		  assert(GPUGenie_knn_DEFAULT_DATA_PER_THREAD % num_per_u32 == 0);
		  u32 total_threads =  d_selected_bitmap.size() / (GPUGenie_knn_DEFAULT_DATA_PER_THREAD / num_per_u32) + 1;

		  augment_bitmap<<<total_threads/GPUGenie_knn_THREADS_PER_BLOCK + 1,GPUGenie_knn_THREADS_PER_BLOCK>>>
				  (d_augmented_bitmap_p, d_selected_bitmap_p, d_selected_bitmap.size(), augmented_size, num_per_u32);
		  //cudaCheckErrors(cudaDeviceSynchronize());
#ifdef GPUGENIE_DEBUG
		  end = getTime();
		  printf(">>>>> augment bitmap takes %fms <<<<<\n", getInterval(start, end));
		  start = getTime();
#endif
		  /**   Debug Section  **/
		  // host_vector<u32> h_augmented_bitmap(d_augmented_bitmap);
		  // printf("The first 2048 items in augmented bitmap are:\n");
		  // for(int i = 0; i < 2048; ++i)
		  // {
			 //  printf("%d, ", h_augmented_bitmap[i]);
		  // }
		  // printf("\n");
		  // host_vector<u32> h_selected_bitmap(d_selected_bitmap);
		  // printf("The first 8 items in selected bitmap are:\n");
		  // for(int i = 0; i < 2; ++i)
		  // {
			 //  printf("%d, ", h_selected_bitmap[i]);
		  // }
		  // printf("\n");
		  /** End of Debug Section **/

		  d_selected_bitmap.clear();
		  device_vector<u32>().swap(d_selected_bitmap);

		  std::vector<GPUGenie::query> selected_queries;

		  //In case queries have different topk settings
		  u32 index_size = 0u;
		  for(u32 i = 0; i < selected_query_index.size(); ++i)
		  {
			selected_queries.push_back(queries[selected_query_index[i]]);
			index_size += selected_queries[i].topk();
		  }

		  topk(d_augmented_bitmap, selected_queries, d_selected_top_indexes, u32(bitmap_threshold));
		  //cudaCheckErrors(cudaDeviceSynchronize());

		  d_augmented_bitmap.clear();
		  device_vector<u32>().swap(d_augmented_bitmap);
		  /**   Debug Section  **/
		  // printf("The items in d_selected_top_indexes are:\n");
		  // for(u32 i = 0; i < selected_query_index.size(); ++i)
		  // {
		  // 	  printf("Query %d:\n", selected_query_index[i]);
		  // 	  for(u32 j = 0; j < selected_queries[i].topk(); ++j)
		  // 	  {
				// printf("%d, ", h_selected_top_indexes[i * selected_queries[i].topk() + j] - i * table.i_size());
		  // 	  }
			 //  printf("\n");
		  // }
		  /** End of Debug Section **/

 //		  for(u32 i = 0; i < selected_query_index.size(); ++i)
 //		  {
 //			  for(u32 j = 0; j < selected_queries[i].topk(); ++j)
 //			  {
 //				 h_selected_top_indexes[i * selected_queries[i].topk() + j] -= table.i_size()*i;
 //			  }
 //		  }
#ifdef GPUGENIE_DEBUG
		  end = getTime();
		  printf(">>>>> selected bitmap topk takes %fms <<<<<\n", getInterval(start, end));
		  start =getTime();
#endif
		  int * d_selected_top_indexes_p = thrust::raw_pointer_cast(d_selected_top_indexes.data());
		  correct_index<<<d_selected_top_indexes.size() / GPUGenie_knn_THREADS_PER_BLOCK + 1, GPUGenie_knn_THREADS_PER_BLOCK>>>
					   (d_selected_top_indexes_p, d_selected_top_indexes.size(), queries[0].topk(), table.i_size());
		  //cudaCheckErrors(cudaDeviceSynchronize());
#ifdef GPUGENIE_DEBUG
		  end = getTime();
		  printf(">>>>>correct selected bitmap index takes %fms <<<<<\n", getInterval(start, end));
#endif
		  /**   Debug Section  **/
 //		   printf("The items in d_selected_top_indexes are:\n");
 //		   for(u32 i = 0; i < selected_query_index.size(); ++i)
 //		   {
 //		   	  printf("Query %d:\n", selected_query_index[i]);
 //		   	  for(u32 j = 0; j < selected_queries[i].topk(); ++j)
 //		   	  {
 //				 printf("%d, ", h_selected_top_indexes[i * selected_queries[i].topk() + j]);
 //		   	  }
 //			   printf("\n");
 //		   }
		  /** End of Debug Section **/
	  }

  }

#ifdef GPUGENIE_DEBUG  //for improve
  u64 endKnn = getTime();
  printf(">>>>> knn() before topk and extractIndex %f ms <<<<<\n", getInterval(startKnn, endKnn));
#endif

#ifdef GPUGENIE_DEBUG
  printf("Start topk....\n");
  u64 start = getTime();
#endif

  topk(d_data, queries, d_top_indexes, float(dim));
  //cudaCheckErrors(cudaDeviceSynchronize());

#ifdef GPUGENIE_DEBUG
  u64 end = getTime();
  printf("Topk Finished! \n");
  printf(">>>>> main topk takes %fms <<<<<\n", getInterval(start, end));
  start=getTime();
#endif

  extract_index<<<d_top_indexes.size() / GPUGenie_knn_THREADS_PER_BLOCK + 1, GPUGenie_knn_THREADS_PER_BLOCK>>>
			   (thrust::raw_pointer_cast(d_top_indexes.data()), thrust::raw_pointer_cast(d_data.data()), d_top_indexes.size());
  //cudaCheckErrors(cudaDeviceSynchronize());

  // If has selected topk results, then overwrite the main vector d_top_indexes with the selected results
//  if(d_selected_top_indexes.size() != 0 && selected_query_index.size() != 0)//for improve
//  {
//	  int qid;
//	  for(u32 i = 0; i < selected_query_index.size(); ++i)
//	  {
//		  qid = selected_query_index[i];
//		  thrust::copy(d_selected_top_indexes.begin()+i*queries[qid].topk(),
//					   d_selected_top_indexes.begin()+(i+1)*queries[qid].topk(),
//					   d_top_indexes.begin()+qid*queries[qid].topk());
//	  }
//  }

#ifdef GPUGENIE_DEBUG
  end=getTime();
  printf("Finish topk search!\n");
  printf(">>>>> extract index and copy selected topk results takes %fms <<<<<\n", getInterval(start, end));
#endif
}
