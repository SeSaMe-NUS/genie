#ifndef GaLG_query_h
#define GaLG_query_h

#include "inv_table.h"

#include <vector>

using namespace std;

namespace GaLG {
  class query {
  private:
    struct dim {
      int low;
      int up;
      float weight;
    };

    enum {
      not_builded,
      builded
    } _build_status;
    inv_table* _ref_table;
    vector<dim> _dims;

  public:
    query(inv_table* ref);
    query(inv_table& ref);
    void attr(int index, int low, int up, float weight);
    void build();
  };
}

#endif