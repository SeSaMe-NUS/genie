#include "topk.h"

struct ValueOf
{
  float max;
  __host__ __device__ float
  valueOf(float data)
  {
    return (float) max - data;
  }
};

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

  ValueOf val;
  val.max = *minmax.second;
  bucket_topk<float, ValueOf>(&d_search, val, *minmax.first, *minmax.second,
      &d_tops, &d_end_index, parts, &d_top_indexes);
}
