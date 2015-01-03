#include "topk.h"

struct ValueOf
{
  __host__ __device__ float
  valueOf(float data)
  {
    return (float) data;
  }
} val;

void
GaLG::topk(device_vector<float>& d_search, device_vector<int>& d_tops,
    device_vector<int>& d_top_indexes)
{
  int parts = d_tops.size();
  int total = 0, i;
  for (i = 0; i < d_tops.size(); i++)
    {
      total += d_tops[i];
    }
  float* min = thrust::min_element(raw_pointer_cast(d_search.data()), raw_pointer_cast(d_search.data()) + d_search.size());
  float* max = thrust::max_element(raw_pointer_cast(d_search.data()), raw_pointer_cast(d_search.data()) + d_search.size());

  device_vector<int> d_end_index(parts);
  int number_of_each = d_search.size() / parts;
  for(i=0; i<parts; i++)
    {
      d_end_index[i] = (i+1) * number_of_each;
    }
  d_top_indexes.clear(), d_top_indexes.resize(total);
  bucket_topk<float, ValueOf>(&d_search, val, *min, *max, &d_tops, &d_end_index, parts, &d_top_indexes);
}
