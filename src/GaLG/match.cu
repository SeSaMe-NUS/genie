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
        float* d_aggregation)
    {
    }
  }
}

void
GaLG::match(inv_table& table, vector<query>& queries,
    device_vector<float>& d_aggregation)
{
  int total = table.i_size() * queries.size();
  d_aggregation.clear(), d_aggregation.resize(total);

  vector<query::dim> dims;
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      queries[i].dump(dims);
    }

  device_vector<int> d_ck(*table.ck());
  int* d_ck_p = raw_pointer_cast(d_ck.data());

  device_vector<int> d_inv(*table.inv());
  int* d_inv_p = raw_pointer_cast(d_inv.data());

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());

  float* d_aggregation_p = raw_pointer_cast(d_aggregation.data());

  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>(table.m_size(), table.i_size(), d_ck_p, d_inv_p, d_dims_p, d_aggregation_p);
}
