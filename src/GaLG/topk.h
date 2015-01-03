#ifndef GaLG_topk_h
#define GaLG_topk_h

#include "GaLG/lib/bucket_topk/bucket_topk.h"
#include <thrust/device_vector.h>
#include <vector>

using namespace std;
using namespace thrust;

namespace GaLG
{
  void
  topk(device_vector<float>& d_search, device_vector<int>& tops,
      device_vector<int>& d_top_indexes);
}

#endif
