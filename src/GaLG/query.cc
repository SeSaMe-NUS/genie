#include "query.h"

GaLG::query::query(inv_table* ref)
{
  _ref_table = ref;
  _dims.clear(); _dims.resize(_ref_table -> m_size());
  int i;
  for(i=0; i<_dims.size(); i++)
  {
    /* Mark always not match */
    _dims[i].low = 0;
    _dims[i].up = -1;
    _dims[i].weight = 0;
  }
  
  _build_status = not_builded;
}

GaLG::query::query(inv_table& ref)
{
  _ref_table = &ref;
  _dims.clear(); _dims.resize(_ref_table -> m_size());
  int i;
  for(i=0; i<_dims.size(); i++)
  {
    /* Mark always not match */
    _dims[i].low = 0;
    _dims[i].up = -1;
    _dims[i].weight = 0;
  }

  _build_status = not_builded;
}

void GaLG::query::attr(int index, int low, int up, float weight)
{
  _build_status = not_builded;

  if(index < 0 || index >= _dims.size())
    return;

  if(low < up)
  {
    /* Mark always not match */
    _dims[index].low = 0;
    _dims[index].up = -1;
    _dims[index].weight = 0;
    return;   
  }

  _dims[index].low = low;
  _dims[index].up = up;
  _dims[index].weight = weight;
}

void GaLG::query::build()
{

}
