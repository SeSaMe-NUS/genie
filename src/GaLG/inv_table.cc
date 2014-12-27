#include "inv_table.h"

#include "inv_list.h"

using namespace GaLG;

void GaLG::inv_table::clear()
{
  _build_status = not_builded;
  _inv_lists.clear();
}

bool GaLG::inv_table::empty()
{
  return _size == -1;
}

void GaLG::inv_table::append(inv_list& inv)
{
  if(_size == -1 || _size == inv.size())
  {
    _build_status = not_builded;
    _size = inv.size();
    _inv_lists.push_back(inv);
  }
}

void GaLG::inv_table::build()
{
  _ck.clear(), _inv.clear();
  int i, j, key, dim, value, last;
  for(i=0; i<_inv_lists.size(); i++)
  {
    dim = i << _value_bit;
    for(value=_inv_lists[i].min(); value<=_inv_lists[i].max(); value++)
    {
      key = dim + value - _inv_lists[i].min();
      vector<int>& index = *_inv_lists[i].index(value);
      
      if(_ck.size() <= key)
      {
        last = _ck.size();
        _ck.resize(key + 1);
        for(; last < _ck.size(); last++)
          _ck[last] = _inv.size();
      }
      for(j=0; j<index.size(); j++)
      {
        _inv.push_back(index[j]);
        _ck[key] = _inv.size();
      }
    }
  }
  _build_status = builded;
}