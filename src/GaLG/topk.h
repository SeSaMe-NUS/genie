#ifndef GaLG_topk_h
#define GaLG_topk_h

#include "GaLG/lib/bucket_topk/bucket_topk.h"
#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

namespace GaLG
{
  /**
   * @brief Find the top k values in given device_vector.
   * @details Find the top k values in given device_vector.
   * 
   * @param d_search The data vector
   * @param d_tops The top k values.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(device_vector<float>& d_search, device_vector<int>& d_tops,
      device_vector<int>& d_top_indexes);
}

#endif
