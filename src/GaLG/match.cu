#include "match.h"

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
#endif

namespace GaLG
{
  namespace device
  {
    __global__
    void
    match(int m_size, int i_size, int* d_ck, int* d_inv, query::dim* d_dims,
        int* d_count, float* d_aggregation)
    {
      query::dim* q = &d_dims[blockIdx.x];

      int min, max;
      min = q->low;
      max = q->up;
      if (min > max)
        return;

      min < 1 ? min = 0 : min = d_ck[min - 1];
      max = d_ck[max];

      int loop = (max - min) / GaLG_device_THREADS_PER_BLOCK + 1;
      int part = blockIdx.x / m_size * i_size;

      int i;
      for (i = 0; i < loop; i++)
        {
          if (threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min < max)
            {
              atomicAdd(
                  &d_count[part]
                      + d_inv[threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK
                          + min], 1);
              atomicAdd(
                  &d_aggregation[part]
                      + d_inv[threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK
                          + min], q->weight);
            }
        }
    }
  }
}

void
GaLG::match(inv_table& table, vector<query>& queries,
    device_vector<int>& d_count, device_vector<float>& d_aggregation)
        throw (int)
{
  if (table.build_status() == inv_table::not_builded)
    throw inv_table::not_builded_exception;

  vector<query::dim> dims;
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      if (queries[i].ref_table() != &table)
        throw inv_table::not_matched_exception;
      if (table.build_status() == inv_table::builded)
        queries[i].build();
      else if (table.build_status() == inv_table::builded_compressed)
        queries[i].build_compressed();
      queries[i].dump(dims);
    }

  int total = table.i_size() * queries.size();

  device_vector<int> d_ck(*table.ck());
  int* d_ck_p = raw_pointer_cast(d_ck.data());

  device_vector<int> d_inv(*table.inv());
  int* d_inv_p = raw_pointer_cast(d_inv.data());

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());

  d_count.clear(), d_count.resize(total);
  int* d_count_p = raw_pointer_cast(d_count.data());

  d_aggregation.clear(), d_aggregation.resize(total);
  float* d_aggregation_p = raw_pointer_cast(d_aggregation.data());

  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
  (table.m_size(), table.i_size(), d_ck_p, d_inv_p, d_dims_p, d_count_p, d_aggregation_p);
}

void
GaLG::match(inv_table& table, query& queries, device_vector<int>& d_count,
    device_vector<float>& d_aggregation) throw (int)
{
  vector<query> _q;
  _q.push_back(queries);
  match(table, _q, d_count, d_aggregation);
}
