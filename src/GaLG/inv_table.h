#ifndef GaLG_inv_table_h
#define GaLG_inv_table_h

#include "raw_data.h"
#include "inv_list.h"

#include <vector>

using namespace std;

namespace GaLG {
  class inv_table {
  private:
    enum {
      not_builded,
      builded
    } _build_status;
    int _value_bit;
    int _size;
    vector<inv_list> _inv_lists;
    vector<int> _ck, _inv;

  public:
    inv_table() : _value_bit(16), _size(-1) {}
    void clear();
    bool empty();
    void append(inv_list&);
    void build();
  };
}

#endif