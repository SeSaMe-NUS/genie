#ifndef GaLG_match_h
#define GaLG_match_h

#include "inv_table.h"
#include "query.h"

#include <vector>

using namespace std;

namespace GaLG {
  void match(inv_table& table, vector<query>& queries, vector<float>& agg);

  namespace device {
    __global__
    void match();
  }
}

#endif