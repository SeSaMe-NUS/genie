#include "query.h"

GaLG::query::query(inv_table* ref)
{
  _ref_table = ref;
  _attr.clear();
  _attr.resize(_ref_table->m_size());
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      _attr[i].low = 0;
      _attr[i].up = -1;
      _attr[i].weight = 0;

      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
  _topk = 1;
}

GaLG::query::query(inv_table& ref)
{
  _ref_table = &ref;
  _attr.clear();
  _attr.resize(_ref_table->m_size());
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      _attr[i].low = 0;
      _attr[i].up = -1;
      _attr[i].weight = 0;

      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
  _topk = 1;
}

GaLG::inv_table*
GaLG::query::ref_table()
{
  return _ref_table;
}

void
GaLG::query::attr(int index, int low, int up, float weight)
{
  if (index < 0 || index >= _dims.size())
    return;

  _attr[index].low = low;
  _attr[index].up = up;
  _attr[index].weight = weight;
}

void
GaLG::query::topk(int k)
{
  _topk = k;
}

int
GaLG::query::topk()
{
  return _topk;
}

void
GaLG::query::build()
{
  int index, low, up;
  float weight;
  for (index = 0; index < _attr.size(); index++)
    {
      low = _attr[index].low;
      up = _attr[index].up;
      weight = _attr[index].weight;

      if (low > up)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;;
        }

      int d = index << _ref_table->shifter();
      vector<inv_list>& inv_lists = *_ref_table->inv_lists();
      inv_list& inv = inv_lists[index];
      _dims[index].weight = weight;

      if (low < inv.min())
        {
          low = inv.min();
        }

      if (low > inv.max())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      if (up > inv.max())
        {
          up = inv.max();
        }

      if (up < inv.min())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      _dims[index].low = d + low - inv.min();
      _dims[index].up = d + up - inv.min();
    }
}

void
GaLG::query::build_compressed()
{
  int index, low, up;
  float weight;
  map<int, int>::iterator lower_bound;

  for (index = 0; index < _attr.size(); index++)
    {
      low = _attr[index].low;
      up = _attr[index].up;
      weight = _attr[index].weight;

      if (low > up)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;;
        }

      int d = index << _ref_table->shifter();
      vector<inv_list>& inv_lists = *_ref_table->inv_lists();
      inv_list& inv = inv_lists[index];
      _dims[index].weight = weight;

      if (low < inv.min())
        {
          low = inv.min();
        }

      if (low > inv.max())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      if (up > inv.max())
        {
          up = inv.max();
        }

      if (up < inv.min())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      lower_bound = _ref_table->ck_map()->lower_bound(d + low - inv.min());
      _dims[index].low = lower_bound->second;

      lower_bound = _ref_table->ck_map()->lower_bound(d + up - inv.min());
      if (lower_bound->first - d + inv.min() != up)
        {
          lower_bound--;
        }
      _dims[index].up = lower_bound->second;

      if (_dims[index].up < _dims[index].low)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }
    }
}

void
GaLG::query::dump(vector<dim>& vout)
{
  int i;
  for (i = 0; i < _dims.size(); i++)
    vout.push_back(_dims[i]);
}
