#include "inv_table.h"
#include "stdio.h"
#include "inv_list.h"
#include "Logger.h"

using namespace GPUGenie;

void
GPUGenie::inv_table::init()
{
    _shifter = 16;
    _size = -1;
    _build_status =  not_builded;
    _inv_lists.clear();
    _ck.clear();
    _inv.clear();
    _inv_index.clear();
}

void
GPUGenie::inv_table::clear()
{
  _build_status = not_builded;
  _inv_lists.clear();
  _ck.clear();
  _inv.clear();
  _ck_map.clear();
  clear_gpu_mem();
}

GPUGenie::inv_table::~inv_table()
{
    if(is_stored_in_gpu == true)
    {
        cudaFree(d_inv_p);
        cudaFree(d_inv_index_p);
        cudaFree(d_inv_pos_p);
        cudaFree(d_ck_p);
    }
}

void
GPUGenie::inv_table::clear_gpu_mem()
{
    if(is_stored_in_gpu == false)
        return;

    cudaFree(d_inv_p);
    cudaFree(d_inv_index_p);
    cudaFree(d_inv_pos_p);
    cudaFree(d_ck_p);
    is_stored_in_gpu = false;

}

bool
GPUGenie::inv_table::empty()
{
  return _size == -1;
}

int
GPUGenie::inv_table::m_size()
{
  return _inv_lists.size();
}

int
GPUGenie::inv_table::i_size()
{
  return _size <= -1 ? 0 : _size;
}

int
GPUGenie::inv_table::shifter()
{
  return _shifter;
}

void
GPUGenie::inv_table::append(inv_list& inv)
{
  if (_size == -1 || _size == inv.size())
    {
      _build_status = not_builded;
      _size = inv.size();
      _inv_lists.push_back(inv);
    }
}

void
GPUGenie::inv_table::append(inv_list* inv)
{
  if (inv != NULL)
    {
      append(*inv);
    }
}

GPUGenie::inv_table::status
GPUGenie::inv_table::build_status()
{
  return _build_status;
}

vector<inv_list>*
GPUGenie::inv_table::inv_lists()
{
  return &_inv_lists;
}

vector<int>*
GPUGenie::inv_table::ck()
{
  return &_ck;
}

vector<int>*
GPUGenie::inv_table::inv()
{
  return &_inv;
}

vector<int>*
GPUGenie::inv_table::inv_index()
{
	return &_inv_index;
}

vector<int>*
GPUGenie::inv_table::inv_pos()
{
	return &_inv_pos;
}

map<int, int>*
GPUGenie::inv_table::ck_map()
{
  return &_ck_map;
}

void
GPUGenie::inv_table::build(u64 max_length)
{
  _ck.clear(), _inv.clear();
  _inv_index.clear(); _inv_pos.clear();
  int i, j, key, dim, value, last;
  for (i = 0; i < _inv_lists.size(); i++)
    {
      dim = i << _shifter;
      for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
        {
          key = dim + value - _inv_lists[i].min();
          vector<int>& index = *_inv_lists[i].index(value);

          if (_ck.size() <= key)
            {
              last = _ck.size();
              _ck.resize(key + 1);
              _inv_index.resize(key+1);
              for (; last < _ck.size(); last++)
              {
            	  _ck[last] = _inv.size();
            	  _inv_index[last] = _inv_pos.size();
              }
            }
          for (j = 0; j < index.size(); j++)
            {
        	  if(j % max_length == 0){
        		  _inv_pos.push_back(_inv.size());
        	  }
              _inv.push_back(index[j]);
              _ck[key] = _inv.size();
            }

        }

    }
  _inv_index.push_back(_inv_pos.size());
  _inv_pos.push_back(_inv.size());

  _build_status = builded;
  Logger::log(Logger::DEBUG, "inv_index size %d:", _inv_index.size());
  Logger::log(Logger::DEBUG, "inv_pos size %d:", _inv_pos.size());
  Logger::log(Logger::DEBUG, "inv size %d:", _inv.size());
}

void
GPUGenie::inv_table::build_compressed()
{
  _ck.clear(), _inv.clear(), _ck_map.clear();
  int i, j, key, dim, value;
  for (i = 0; i < _inv_lists.size(); i++)
    {
      dim = i << _shifter;
      for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
        {
          key = dim + value - _inv_lists[i].min();
          vector<int>* indexes = _inv_lists[i].index(value);

          for (j = 0; j < indexes->size(); j++)
            {
              _inv.push_back((*indexes)[j]);
              _ck_map[key] = _ck.size();
            }
          if (indexes->size() > 0)
            {
              _ck.push_back(_inv.size());
            }
        }
    }
  _build_status = builded_compressed;
}
/*

void
GPUGenie::inv_table::serialize_to_file(const char* filename)
{
    ofsream ofs(filename, ios::trunc|ios::binary);
    if(!ofs.is_open())
    {
        return;
    }
    int status_int = _build_status;
    ofs.write((char*)&status_int, sizeof(int));
    ofs.write((char*)&_shifter, sizeof(int));
    ofs.write((char*)&_size, sizeof(int));


}

void
GPUGenie::inv_table::deserialize_from_file(const char* filename)
{
     
}
*/
