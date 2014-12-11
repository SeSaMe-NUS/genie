#include "raw_data.h"

#include <vector>
#include <string>

using namespace std;

int GaLG::raw_data::m_size()
{
  return meta.size();
}

int GaLG::raw_data::i_size()
{
  return instance.size();
}

GaLG::raw_data& GaLG::raw_data::select(string attr, vector<string>& output)
{
  int attr_number;
  for(attr_number = 0; attr_number < m_size(); attr_number++)
    if(meta[attr_number].compare(attr) == 0) break;
  select(attr_number, output);
  return *this;
}

GaLG::raw_data& GaLG::raw_data::select(int attr_index, vector<string>& output)
{
  if(attr_index >= m_size() || attr_index < 0)
    throw -1;

  output.clear();
  int i;
  for(i = 0; i < i_size(); i++)
  {
    output.push_back(instance[i][attr_index]);
  }
  return *this;
}