#include "inv_list.h"

int GaLG::inv_list::min()
{
  return bound.first;
}

int GaLG::inv_list::max()
{
  return bound.second;
}