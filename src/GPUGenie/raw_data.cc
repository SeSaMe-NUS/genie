#include "raw_data.h"

#include <vector>
#include <string>

using namespace std;

void
GPUGenie::raw_data::clear()
{
  _meta.clear();
  _instance.clear();
}

int
GPUGenie::raw_data::m_size()
{
  return _meta.size();
}

int
GPUGenie::raw_data::i_size()
{
  return _instance.size();
}

string*
GPUGenie::raw_data::meta(int index)
{
  if (index >= m_size() || index < 0)
    return NULL;
  string* _meta_name = &_meta[index];
  return _meta_name;
}

int
GPUGenie::raw_data::meta(string attr)
{
  int attr_number;
  for (attr_number = 0; attr_number < m_size(); attr_number++)
    if (_meta[attr_number].compare(attr) == 0)
      break;
  if (attr_number == m_size())
    attr_number = -1;
  return attr_number;
}

vector<string>*
GPUGenie::raw_data::row(int index)
{
  if (index >= i_size() || index < 0)
    return NULL;
  return &_instance[index];
}

vector<string>*
GPUGenie::raw_data::col(string attr)
{
  int attr_number;
  for (attr_number = 0; attr_number < m_size(); attr_number++)
    if (_meta[attr_number].compare(attr) == 0)
      break;
  return col(attr_number);
}

vector<string>*
GPUGenie::raw_data::col(int attr_index)
{
  if (attr_index >= m_size() || attr_index < 0)
    return NULL;

  if (_transpose.size() != m_size())
    _transpose.resize(m_size());

  if (!_transpose[attr_index].empty())
    return &_transpose[attr_index];

  _transpose[attr_index].clear();
  int i;
  for (i = 0; i < i_size(); i++)
    {
      _transpose[attr_index].push_back(_instance[i][attr_index]);
    }
  return &_transpose[attr_index];
}
