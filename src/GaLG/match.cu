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
    match()
    {
    }
  }
}

void
GaLG::match(inv_table& table, vector<query>& queries, device_vector<float>& d_agg)
{
  int total = table.i_size() * queries.size();
  d_agg.clear(), d_agg.resize(total);

  vector<query::dim> dims;
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      queries[i].dump(dims);
    }

  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>();
}
