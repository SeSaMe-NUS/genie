#include "topk.h"
#include "GaLG/lib/bucket_topk/bucket_topk.h"

#include <thrust/host_vector.h>
#include <thrust/extrema.h>

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

void
GaLG::topk(GaLG::inv_table& table, GaLG::query& queries,
    device_vector<int>& d_top_indexes)
{
  device_vector<int> d_c;
  device_vector<float> d_a;
  vector<query> q;
  q.push_back(queries);
  match(table, q, d_c, d_a);
  topk(d_a, q, d_top_indexes);
}

void
GaLG::topk(GaLG::inv_table& table, vector<GaLG::query>& queries,
    device_vector<int>& d_top_indexes)
{
  device_vector<int> d_c;
  device_vector<float> d_a;
  match(table, queries, d_c, d_a);
  topk(d_a, queries, d_top_indexes);
}

void
GaLG::topk(device_vector<int>& d_search, vector<GaLG::query>& queries,
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
GaLG::topk(device_vector<float>& d_search, vector<GaLG::query>& queries,
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
GaLG::topk(device_vector<int>& d_search, device_vector<int>& d_tops,
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
GaLG::topk(device_vector<float>& d_search, device_vector<int>& d_tops,
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
