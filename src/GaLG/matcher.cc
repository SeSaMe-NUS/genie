#include "matcher.h"

bool GaLG::matcher::unset::match(int value)
{
  return false;
}

GaLG::matcher::range::range(int low, int up)
{
  _low = low, _up = up;
}

bool GaLG::matcher::range::match(int value)
{
  return _low <= value && value <= _up;
}