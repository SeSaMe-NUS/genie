#ifndef GaLG_match_h
#define GaLG_match_h

#include "inv_table.h"
#include "query.h"

#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

namespace GaLG
{
  void
  match(inv_table& table, vector<query>& queries, device_vector<float>& d_agg);
}

#endif
