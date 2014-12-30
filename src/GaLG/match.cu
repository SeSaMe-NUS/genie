#include "match.h"

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
#endif

void GaLG::match(inv_table& table, vector<query>& queries, vector<float>& agg)
{
  int total = table.i_size() * queries.size();
  agg.clear(), agg.resize(total);

  vector<query::dim> dims;
  int i;
  for(i=0; i<queries.size(); i++)
  {
    queries[i].dump(dims);
  }

  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>();
}

__global__
void GaLG::device::match();