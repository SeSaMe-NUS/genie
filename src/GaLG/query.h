#ifndef GaLG_query_h
#define GaLG_query_h

#include "inv_table.h"
#include "inv_list.h"

#include <vector>

using namespace std;

namespace GaLG
{
  class query
  {
  public:
    struct dim
    {
      int low;
      int up;
      float weight;
    };
  private:
    inv_table* _ref_table;
    vector<dim> _attr;
    vector<dim> _dims;

  public:
    query(inv_table* ref);
    query(inv_table& ref);
    inv_table*
    ref_table();
    void
    attr(int index, int low, int up, float weight);

    void
    build();

    void
    build_compressed();

    void
    dump(vector<dim>& vout);
  };
}

#endif
