#include "inv_list.h"

#include <cstdlib>

int
cv(string& s, void* d)
{
  return atoi(s.c_str());
}

GPUGenie::inv_list::inv_list(vector<int>& vin)
{
  invert(vin);
}

GPUGenie::inv_list::inv_list(vector<int>* vin)
{
  invert(vin);
}

GPUGenie::inv_list::inv_list(vector<string>& vin)
{
  invert(vin);
}

GPUGenie::inv_list::inv_list(vector<string>* vin)
{
  invert(vin);
}

int
GPUGenie::inv_list::min()
{
  return _bound.first;
}

int
GPUGenie::inv_list::max()
{
  return _bound.second;
}

int
GPUGenie::inv_list::size()
{
  return _size;
}
void
GPUGenie::inv_list::invert_bijectMap(vector<vector<int> > & vin)
{
	  _size = vin.size();
	  if (vin.empty())
	    return;

	  _bound.first = vin[0][0], _bound.second = vin[0][0], _inv.clear();

	  unsigned int i,j;
	  for (i = 0; i < vin.size(); i++)
	  {
		  for(j = 0; j < vin[i].size(); ++j)
		  {
		      if (_bound.first > vin[i][j])
		        _bound.first = vin[i][j];
		      if (_bound.second < vin[i][j])
		        _bound.second = vin[i][j];
		  }
	  }
	  unsigned int gap = _bound.second - _bound.first + 1;
	  _inv.resize(gap);
	  for (i = 0; i < gap; i++)
	    _inv[i].clear();

	  for (i = 0; i < vin.size(); i++)
	  {
		  for(j = 0; j < vin[i].size(); ++j)
		  {
			  _inv[vin[i][j] - _bound.first].push_back(i);
		  }
	  }
	  return;
}

void
GPUGenie::inv_list::invert_bijectMap(int *data, unsigned int item_num, unsigned int *index, unsigned int row_num)
{
    _size = row_num;
    if(item_num == 0)
        return;
    unsigned int i,j;
    for(i=0 ; i<item_num; ++i){
        if(_bound.first > data[i])
            _bound.first = data[i];
        if(_bound.second < data[i])
            _bound.second = data[i];
    }
    unsigned int gap = _bound.second - _bound.first + 1;
    _inv.resize(gap);
    for( i=0 ; i<gap ; ++i)
        _inv[i].clear();
    for(i=0; i<row_num-1 ; ++i)
        for(j=index[i]; j<index[i+1]; ++j)
            _inv[data[j]-_bound.first].push_back(i);

    for(i=index[row_num-1]; i<item_num; ++i)
        _inv[data[i]-_bound.first].push_back(row_num-1);
    cout<<"p6"<<endl;
    return;
}





void
GPUGenie::inv_list::invert(vector<int>& vin)
{
  _size = vin.size();
  if (vin.empty())
    return;

  _bound.first = vin[0], _bound.second = vin[0], _inv.clear();

  unsigned int i;
  for (i = 0; i < vin.size(); i++)
    {
      if (_bound.first > vin[i])
        _bound.first = vin[i];
      if (_bound.second < vin[i])
        _bound.second = vin[i];
    }

  unsigned int gap = _bound.second - _bound.first + 1;
  _inv.resize(gap);
  for (i = 0; i < gap; i++)
    _inv[i].clear();

  for (i = 0; i < vin.size(); i++)
    _inv[vin[i] - _bound.first].push_back(i);
  return;
}

void
GPUGenie::inv_list::invert(vector<int>* vin)
{
  invert(*vin);
}

void
GPUGenie::inv_list::invert(vector<string>& vin)
{
  invert(vin, &cv, NULL);
}

void
GPUGenie::inv_list::invert(vector<string>* vin)
{
  invert(*vin);
}

void
GPUGenie::inv_list::invert(vector<string>& vin, int
(*stoi)(string&, void*), void* d)
{
  _size = vin.size();
  if (vin.empty())
    return;

  _bound.first = stoi(vin[0], d);
  _bound.second = stoi(vin[0], d);
  _inv.clear();

  unsigned int i;
  for (i = 0; i < vin.size(); i++)
    {
      if (_bound.first > stoi(vin[i], d))
        _bound.first = stoi(vin[i], d);
      if (_bound.second < stoi(vin[i], d))
        _bound.second = stoi(vin[i], d);
    }

  unsigned int gap = _bound.second - _bound.first + 1;
  _inv.resize(gap);
  for (i = 0; i < gap; i++)
    _inv[i].clear();

  for (i = 0; i < vin.size(); i++)
    _inv[stoi(vin[i], d) - _bound.first].push_back(i);
  return;
}


void
GPUGenie::inv_list::invert(vector<string>* vin, int
(*stoi)(string&, void*), void* d)
{
  invert(*vin, stoi, d);
}

bool
GPUGenie::inv_list::contains(int value)
{
  if (value > _bound.second || value < _bound.first)
    return false;
  return true;
}

vector<int>*
GPUGenie::inv_list::index(int value)
{
  if (!contains(value))
    return NULL;
  return &_inv[value - min()];
}
