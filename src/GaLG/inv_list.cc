#include "inv_list.h"

#include <cstdlib>

int cv(string& s, void* d)
{
  return atoi(s.c_str());
}

int GaLG::inv_list::min()
{
  return _bound.first;
}

int GaLG::inv_list::max()
{
  return _bound.second;
}

void GaLG::inv_list::invert(vector<int>& vin)
{
  if(vin.empty())
    return;

  _bound.first = vin[0], _bound.second = vin[0], _inv.clear();

  int i;
  for(i = 0; i < vin.size(); i++)
  {
    if(_bound.first > vin[i])
      _bound.first = vin[i];
    if(_bound.second < vin[i])
      _bound.second = vin[i];
  }

  int gap = _bound.second - _bound.first + 1;
  _inv.resize(gap);
  for (i = 0; i < gap; i++)
    _inv[i].clear();

  for (i = 0; i < vin.size(); i++)
    _inv[vin[i] - _bound.first].push_back(i);
  return;
}

void GaLG::inv_list::invert(vector<string>& vin)
{
  invert(vin, &cv, NULL);
}

void GaLG::inv_list::invert(vector<string>& vin, int(*stoi)(string&, void*), void* d)
{
  if(vin.empty())
    return;

  _bound.first = stoi(vin[0], d);
  _bound.second = stoi(vin[0], d);
  _inv.clear();

  int i;
  for(i = 0; i < vin.size(); i++)
  {
    if(_bound.first > stoi(vin[i], d))
      _bound.first = stoi(vin[i], d);
    if(_bound.second < stoi(vin[i], d))
      _bound.second = stoi(vin[i], d);
  }

  int gap = _bound.second - _bound.first + 1;
  _inv.resize(gap);
  for (i = 0; i < gap; i++)
    _inv[i].clear();

  for (i = 0; i < vin.size(); i++)
    _inv[stoi(vin[i], d) - _bound.first].push_back(i);
  return;
}

bool GaLG::inv_list::contains(int value)
{
  if(value > _bound.second || value < _bound.first)
    return false;
  return true;
}

vector<int>* GaLG::inv_list::index(int value)
{
  if(!contains(value))
    return NULL;
  return &_inv[value - min()];
}