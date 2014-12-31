#include "query.h"

GaLG::query::query(inv_table* ref)
{
  _ref_table = ref;
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
}

GaLG::query::query(inv_table& ref)
{
  _ref_table = &ref;
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
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

  if (low > up)
    {
      /* Mark always not match */
      _dims[index].low = 0;
      _dims[index].up = -1;
      return;
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
    }

  _dims[index].low = d + low - inv.min();
  _dims[index].up = d + up - inv.min();
}

void
GaLG::query::dump(vector<dim>& vout)
{
  int i;
  for (i = 0; i < _dims.size(); i++)
    vout.push_back(_dims[i]);
}
