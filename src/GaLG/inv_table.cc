#include "inv_table.h"

#include "inv_list.h"

using namespace GaLG;

void
GaLG::inv_table::clear()
{
  _build_status = not_builded;
  _inv_lists.clear();
  _ck.clear();
  _inv.clear();
  _ck_map.clear();
}

bool
GaLG::inv_table::empty()
{
  return _size == -1;
}

int
GaLG::inv_table::m_size()
{
  return _inv_lists.size();
}

int
GaLG::inv_table::i_size()
{
  return _size <= -1 ? 0 : _size;
}

int
GaLG::inv_table::shifter()
{
  return _shifter;
}

void
GaLG::inv_table::append(inv_list& inv)
{
  if (_size == -1 || _size == inv.size())
    {
      _build_status = not_builded;
      _size = inv.size();
      _inv_lists.push_back(inv);
    }
}

void
GaLG::inv_table::append(inv_list* inv)
{
  if (inv != NULL)
    {
      append(*inv);
    }
}

GaLG::inv_table::status
GaLG::inv_table::build_status()
{
  return _build_status;
}

vector<inv_list>*
GaLG::inv_table::inv_lists()
{
  return &_inv_lists;
}

vector<int>*
GaLG::inv_table::ck()
{
  return &_ck;
}

vector<int>*
GaLG::inv_table::inv()
{
  return &_inv;
}

map<int, int>*
GaLG::inv_table::ck_map()
{
  return &_ck_map;
}

void
GaLG::inv_table::build()
{
  _ck.clear(), _inv.clear();
  int i, j, key, dim, value, last;
  for (i = 0; i < _inv_lists.size(); i++)
    {
      dim = i << _shifter;
      for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
        {
          key = dim + value - _inv_lists[i].min();
          vector<int>& index = *_inv_lists[i].index(value);

          if (_ck.size() <= key)
            {
              last = _ck.size();
              _ck.resize(key + 1);
              for (; last < _ck.size(); last++)
                _ck[last] = _inv.size();
            }
          for (j = 0; j < index.size(); j++)
            {
              _inv.push_back(index[j]);
              _ck[key] = _inv.size();
            }
        }

    }
  _build_status = builded;
}

void
GaLG::inv_table::build_compressed()
{
  _ck.clear(), _inv.clear(), _ck_map.clear();
  int i, j, key, dim, value;
  for (i = 0; i < _inv_lists.size(); i++)
    {
      dim = i << _shifter;
      for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
        {
          key = dim + value - _inv_lists[i].min();
          vector<int>* indexes = _inv_lists[i].index(value);

          for (j = 0; j < indexes->size(); j++)
            {
              _inv.push_back((*indexes)[j]);
              _ck_map[key] = _ck.size();
            }
          if (indexes->size() > 0)
            {
              _ck.push_back(_inv.size());
            }
        }
    }
  _build_status = builded_compressed;
}
