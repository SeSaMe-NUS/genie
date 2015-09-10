#include "knn.h"

bool GALG_ERROR = false;
unsigned long long GALG_TIME = 0ull;

#ifndef GaLG_knn_THREADS_PER_BLOCK
#define GaLG_knn_THREADS_PER_BLOCK 1024
#endif

#ifndef GaLG_knn_DEFAULT_HASH_TABLE_SIZE
#define GaLG_knn_DEFAULT_HASH_TABLE_SIZE 1
#endif

#ifndef GaLG_knn_DEFAULT_BITMAP_BITS
#define GaLG_knn_DEFAULT_BITMAP_BITS 2
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

void
GaLG::knn(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes)
{
	int hash_table_size = GaLG_knn_DEFAULT_HASH_TABLE_SIZE * table.i_size() + 1;
	knn(table, queries, d_top_indexes, hash_table_size, GaLG_knn_DEFAULT_BITMAP_BITS);
}

void
GaLG::knn(GaLG::inv_table& table, vector<GaLG::query>& queries,
    device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits)
{
	knn(table, queries, d_top_indexes, hash_table_size, bitmap_bits, table.m_size(), 0, 0);
}


void
GaLG::knn(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
		   int bitmap_bits,
		   int dim)
{
	knn(table, queries, d_top_indexes, hash_table_size, bitmap_bits, dim, 0,0);
}
void
GaLG::knn_tweets(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
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
  knn(table, queries, d_top_indexes, hash_table_size, bitmap_bits,
		  	  float(qmax+1), num_of_hot_dims, hot_dim_threshold);
}
void
GaLG::knn(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes,
		   int hash_table_size,
		   int bitmap_bits,
		   int dim,
		   int num_of_hot_dims,
		   int hot_dim_threshold)
{

  printf("Parameters: %d,%d,%d,%d,%d\n", hash_table_size, bitmap_bits, dim, num_of_hot_dims, hot_dim_threshold);
  device_vector<data_t> d_data;
  device_vector<u32> d_bitmap;
  device_vector<u32> d_augmented_bitmap;
  try{
	  match(table, queries, d_data, d_bitmap, hash_table_size, bitmap_bits, num_of_hot_dims, hot_dim_threshold);
	  cudaCheckErrors(cudaDeviceSynchronize());
	  printf("Start topk....\n");
	  topk(d_data, queries, d_top_indexes, float(dim));
	  cudaCheckErrors(cudaDeviceSynchronize());
	  printf("Topk Finished! \n");
	  extract_index<<<d_top_indexes.size() / GaLG_knn_THREADS_PER_BLOCK + 1, GaLG_knn_THREADS_PER_BLOCK>>>
				   (thrust::raw_pointer_cast(d_top_indexes.data()), thrust::raw_pointer_cast(d_data.data()), d_top_indexes.size());
	  cudaCheckErrors(cudaDeviceSynchronize());
  }catch(MemException& e){
	  printf("%s.\n", e.what());
	  printf("Please try again with smaller data/query/hashtable size.\n");
	  GALG_ERROR = true;
  }
  printf("Finish topk search!\n");
}
