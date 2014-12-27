#include "query.h"

GaLG::query::query(inv_table& table) : _dim(table.m_size()), _inv_table(table)
{
  int i;
  for(i=0; i<table.m_size(); i++)
  {
    _dim[i] = new GaLG::matcher::unset();
  }
}

GaLG::query::query(inv_table* table) : _dim(table -> m_size()), _inv_table(*table)
{
  int i;
  for(i=0; i<table -> m_size(); i++)
  {
    _dim[i] = new GaLG::matcher::unset();
  }
}

GaLG::query::~query()
{
  int i;
  for(i=0; i<_dim.size(); i++)
  {
    delete _dim[i];
  }
}

void GaLG::query::dim(int index, dim_matcher* _dim_matcher)
{
  if(index<0 || index>=_dim.size())
    return;
  if(_dim[index] != NULL)
  {
    delete _dim[index];
    _dim[index] = NULL;
  }
  _dim[index] = _dim_matcher;
}