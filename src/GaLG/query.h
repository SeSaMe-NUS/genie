#ifndef GaLG_query_h
#define GaLG_query_h

#include "inv_table.h"
#include "matcher.h"

#include <vector>

using namespace std;

namespace GaLG {
  class query {
  private:
    inv_table& _inv_table;
    vector<dim_matcher*> _dim;

  public:
    query(inv_table&);
    query(inv_table*);
    ~query();
    void dim(int, dim_matcher*);
  };
}

#endif