#include "topk.h"
#include "GaLG/lib/bucket_topk/bucket_topk.h"
#include "match.h"
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#ifndef GaLG_topk_THREADS_PER_BLOCK
#define GaLG_topk_THREADS_PER_BLOCK 1024
#endif

#ifndef GaLG_topk_DEFAULT_HASH_TABLE_SIZE
#define GaLG_topk_DEFAULT_HASH_TABLE_SIZE 0.5
#endif

struct ValueOfFloat
{
  float max;__host__ __device__ float
  valueOf(float data)
  {
    return (float) max - data;
  }
};

struct ValueOfInt
{
  float max;__host__ __device__ float
  valueOf(int data)
  {
    return (float) max - data;
  }
};


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
	id[tId] = od[id[tId]].id;
}

void
GaLG::topk(GaLG::inv_table& table, GaLG::query& queries,
    device_vector<int>& d_top_indexes)
{
	int hash_table_size = GaLG_topk_DEFAULT_HASH_TABLE_SIZE * table.i_size() + 1;
	topk(table, queries, d_top_indexes, hash_table_size);
}

void
GaLG::topk(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes)
{
	int hash_table_size = GaLG_topk_DEFAULT_HASH_TABLE_SIZE * table.i_size() + 1;
	topk(table, queries, d_top_indexes, hash_table_size);
}

void
GaLG::topk(GaLG::inv_table& table, GaLG::query& queries,
    device_vector<int>& d_top_indexes, int hash_table_size)
{
  device_vector<float> d_a;
  device_vector<data_t> d_data;
  vector<query> q;
  q.push_back(queries);
  match(table, q, d_data, hash_table_size);
  d_a.resize(hash_table_size * q.size());
  convert_data<<<hash_table_size * q.size() / GaLG_topk_THREADS_PER_BLOCK + 1, GaLG_topk_THREADS_PER_BLOCK>>>
		  	  (thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_data.data()), hash_table_size * q.size());
  topk(d_a, q, d_top_indexes);
  extract_index<<<d_top_indexes.size() / GaLG_topk_THREADS_PER_BLOCK + 1, GaLG_topk_THREADS_PER_BLOCK>>>
		  	   (thrust::raw_pointer_cast(d_top_indexes.data()), thrust::raw_pointer_cast(d_data.data()), d_top_indexes.size());
}

void
GaLG::topk(GaLG::inv_table& table,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes, int hash_table_size)
{
  device_vector<float> d_a(hash_table_size * queries.size());
  device_vector<data_t> d_data;
  match(table, queries, d_data, hash_table_size);

  printf("Start converting data for topk...\n");
  convert_data<<<hash_table_size * queries.size() / GaLG_topk_THREADS_PER_BLOCK + 1, GaLG_topk_THREADS_PER_BLOCK>>>
		  	  (thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_data.data()), hash_table_size * queries.size());
  cudaCheckErrors(cudaDeviceSynchronize());
  printf("Start topk....\n");
  topk(d_a, queries, d_top_indexes);
  cudaCheckErrors(cudaDeviceSynchronize());
  printf("Topk Finished! \n");
  printf("Start extracting index....\n");
  extract_index<<<d_top_indexes.size() / GaLG_topk_THREADS_PER_BLOCK + 1, GaLG_topk_THREADS_PER_BLOCK>>>
		  	   (thrust::raw_pointer_cast(d_top_indexes.data()), thrust::raw_pointer_cast(d_data.data()), d_top_indexes.size());
  cudaCheckErrors(cudaDeviceSynchronize());
  printf("Finish topk search!\n");
}

void
GaLG::topk(device_vector<int>& d_search,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes)
{
  host_vector<int> h_tops(queries.size());
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      h_tops[i] = queries[i].topk();
    }
  device_vector<int> d_tops(h_tops);
  topk(d_search, d_tops, d_top_indexes);
}

void
GaLG::topk(device_vector<float>& d_search,
		   vector<GaLG::query>& queries,
		   device_vector<int>& d_top_indexes)
{
  host_vector<int> h_tops(queries.size());
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      h_tops[i] = queries[i].topk();
    }
  device_vector<int> d_tops(h_tops);
  topk(d_search, d_tops, d_top_indexes);
}

void
GaLG::topk(device_vector<int>& d_search,
		   device_vector<int>& d_tops,
		   device_vector<int>& d_top_indexes)
{
  int parts = d_tops.size();
  int total = 0, i, num;
  for (i = 0; i < parts; i++)
    {
      num = d_tops[i];
      total += num;
    }
  thrust::pair<device_vector<int>::iterator, device_vector<int>::iterator> minmax =
      thrust::minmax_element(d_search.begin(), d_search.end());
  host_vector<int> h_end_index(parts);
  device_vector<int> d_end_index(parts);
  int number_of_each = d_search.size() / parts;
  for (i = 0; i < parts; i++)
    {
      h_end_index[i] = (i + 1) * number_of_each;
    }
  d_end_index = h_end_index;
  d_top_indexes.clear(), d_top_indexes.resize(total);

  ValueOfInt val;
  val.max = *minmax.second;
  float min = *minmax.first;
  float max = *minmax.second;
  bucket_topk<int, ValueOfInt>(&d_search, val, min, max, &d_tops, &d_end_index,
      parts, &d_top_indexes);
}

void
GaLG::topk(device_vector<float>& d_search,
		   device_vector<int>& d_tops,
		   device_vector<int>& d_top_indexes)
{
  int parts = d_tops.size();
  int total = 0, i, num;
  for (i = 0; i < parts; i++)
    {
      num = d_tops[i];
      total += num;
    }
  thrust::pair<device_vector<float>::iterator, device_vector<float>::iterator> minmax =
      thrust::minmax_element(d_search.begin(), d_search.end());
  host_vector<int> h_end_index(parts);
  device_vector<int> d_end_index(parts);
  int number_of_each = d_search.size() / parts;
  for (i = 0; i < parts; i++)
    {
      h_end_index[i] = (i + 1) * number_of_each;
    }
  d_end_index = h_end_index;
  d_top_indexes.clear(), d_top_indexes.resize(total);

  ValueOfFloat val;
  val.max = *minmax.second;
  bucket_topk<float, ValueOfFloat>(&d_search, val, *minmax.first,
      *minmax.second, &d_tops, &d_end_index, parts, &d_top_indexes);
}
