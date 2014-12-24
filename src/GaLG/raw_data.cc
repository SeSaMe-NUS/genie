#include "raw_data.h"

#include <vector>
#include <string>

using namespace std;

void GaLG::raw_data::clear()
{
  meta.clear();
  instance.clear();
}

int GaLG::raw_data::set_meta(vector<string>& _meta)
{
  clear();
  meta.resize(_meta.size());
  copy(_meta.begin(), _meta.end(), meta.begin());
  return 0;
}

int GaLG::raw_data::set_meta(int index, string new_name)
{
  if(index >= m_size() || index < 0)
    return -1;
  meta[index] = new_name;
  return index;
}

int GaLG::raw_data::set_meta(string old_name, string new_name)
{
  int index;
  for(index = 0; index < m_size(); index++)
    if(meta[index] == old_name)
      break;
  return set_meta(index, new_name);
}

int GaLG::raw_data::get_meta(int index, string& meta_name)
{
  if(index >= m_size() || index < 0)
    return -1;
  meta_name = meta[index];
  return index;
}

string* GaLG::raw_data::get_meta(int index)
{
  if(index >= m_size() || index < 0)
    return NULL;
  string* meta_name = &meta[index];
  return meta_name;
}

int GaLG::raw_data::m_size()
{
  return meta.size();
}

int GaLG::raw_data::i_size()
{
  return instance.size();
}

int GaLG::raw_data::add_row(vector<string>& row)
{
  if(row.size() != m_size())
    return -1;
  instance.push_back(row);
  return instance.size();
}

int GaLG::raw_data::get_row(int index, vector<string>& row)
{
  if(index >= i_size() || index < 0)
    return -1;
  row.clear(), row.resize(instance[index].size());
  copy(instance[index].begin(), instance[index].end(), row.begin());
  return index;
}

vector<string>* GaLG::raw_data::get_row(int index)
{
  if(index >= i_size() || index < 0)
    return NULL;
  return &instance[index];
}

int GaLG::raw_data::select(string attr, vector<string>& output)
{
  int attr_number;
  for(attr_number = 0; attr_number < m_size(); attr_number++)
    if(meta[attr_number].compare(attr) == 0) break;
  return select(attr_number, output);
}

int GaLG::raw_data::select(int attr_index, vector<string>& output)
{
  vector<string>* re = select(attr_index);

  if(re == NULL)
    return -1;

  output.clear(), output.resize(re -> size());
  copy(re -> begin(), re -> end(), output.begin());
  return attr_index;
}

vector<string>* GaLG::raw_data::select(string attr)
{
  int attr_number;
  for(attr_number = 0; attr_number < m_size(); attr_number++)
    if(meta[attr_number].compare(attr) == 0) break;
  return select(attr_number);
}

vector<string>* GaLG::raw_data::select(int attr_index)
{
  if(attr_index >= m_size() || attr_index < 0)
    return NULL;

  if(transpose.size() != m_size())
    transpose.resize(m_size());

  if(!transpose[attr_index].empty())
    return &transpose[attr_index];

  transpose[attr_index].clear();
  int i;
  for(i = 0; i < i_size(); i++)
  {
    transpose[attr_index].push_back(instance[i][attr_index]);
  }
  return &transpose[attr_index];
}