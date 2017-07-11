/*! \file inv_table.cu
 *  \brief Implementation of class inv_table 
 *  declared in file inv_table.h
 */


#include <stdio.h>
#include <fstream>
#include <exception>
#include <iostream>
#include <utility>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp>

#include <genie/utility/cuda_macros.h>
#include <genie/utility/Logger.h>
#include <genie/utility/Timing.h>
#include <genie/exception/exception.h>

#include "inv_table.h"

using namespace std;
using namespace genie::utility;

//int*  inv_table::d_inv_p = NULL;
int genie::table::inv_table::max_inv_size = 0;

bool genie::table::inv_table::cpy_data_to_gpu()
{
	try
	{
		if(d_inv_p == NULL)
			cudaCheckErrors(cudaMalloc(&d_inv_p, sizeof(int) * _inv.size()));
		cudaCheckErrors(cudaMemcpy(d_inv_p, &_inv[0], sizeof(int) * _inv.size(), cudaMemcpyHostToDevice));
		is_stored_in_gpu = true;
	}
	catch(std::bad_alloc &e)
	{
		throw(genie::exception::gpu_bad_alloc(e.what()));
	}

	return true;
}

void genie::table::inv_table::clear()
{
	_build_status = not_builded;
	_inv_lists.clear();
	_ck.clear();
	_inv.clear();
}

genie::table::inv_table::~inv_table()
{
    if(d_inv_p != NULL)
    {

    cout << "Program end ---- cudaFreeTime: " ;
	u64 t1 = getTime();
        cudaCheckErrors(cudaFree(d_inv_p));
        d_inv_p = NULL;

    u64 t2 = getTime();
    cout << getInterval(t1, t2) << " ms."<< endl;
    }
}
void genie::table::inv_table::clear_gpu_mem()
{
	if (d_inv_p == NULL)
		return;

    cout << "cudaFreeTime: " ;
	u64 t1 = getTime();
    cudaCheckErrors(cudaFree(d_inv_p));
    u64 t2 = getTime();
    cout << getInterval(t1, t2) << " ms."<< endl;
}

bool genie::table::inv_table::empty()
{
	return _size == -1;
}

int genie::table::inv_table::m_size()
{
    return _dim_size;
	//return _inv_lists.size();
}

int genie::table::inv_table::i_size()
{
	return _size <= -1 ? 0 : _size;
}

int genie::table::inv_table::shifter()
{
	return _shifter;
}

unordered_map<int, int>* genie::table::inv_table::get_distinct_map(int dim)
{
    return &_distinct_map[dim];
}

void genie::table::inv_table::append(inv_list& inv)
{
	if (_size == -1 || _size == inv.size())
	{
		_build_status = not_builded;
		_size = inv.size();
		_inv_lists.push_back(inv);
        
        _dim_size = _inv_lists.size();
        
        //get size for every posting list

        vector<int> line;
        for(int i = 0 ; i < inv.value_range() ; ++i)
            line.push_back(inv.index(i+inv.min())->size());
        posting_list_size.push_back(line);

        //get upperbound and lowerbound for every inv_list
        inv_list_upperbound.push_back(inv.max());
        inv_list_lowerbound.push_back(inv.min());
	}
}

void genie::table::inv_table::append_sequence(inv_list& inv)
{
    // As for now, we built all data sequence together. Therefore, the _distanct_map is always of size 1
    if(_size == -1)
        _size = 0;
    _build_status = not_builded;
    _size += inv.size();
    _inv_lists.push_back(inv);
    _dim_size = _inv_lists.size();
    
    //get size for every posting list

    vector<int> line;
    for(int i = 0 ; i < inv.value_range() ; ++i)
        line.push_back(inv.index(i+inv.min())->size());
    posting_list_size.push_back(line);

    //get upperbound and lowerbound for every inv_list
    inv_list_upperbound.push_back(inv.max());
    inv_list_lowerbound.push_back(inv.min());
    //distinct_value.push_back(inv.distinct_value_sequence);
	_distinct_map.push_back(inv._distinct);

}

void genie::table::inv_table::append(inv_list* inv)
{
	if (inv != NULL)
	{
		append(*inv);
	}
}

int
genie::table::inv_table::get_posting_list_size(int attr_index, int value)
{
    if((unsigned int)attr_index<posting_list_size.size() && value>=inv_list_lowerbound[attr_index] && value<=inv_list_upperbound[attr_index])
        return posting_list_size[attr_index][value-inv_list_lowerbound[attr_index]];
    else
        return 0;
}

bool
genie::table::inv_table::list_contain(int attr_index, int value)
{
    if(value <= inv_list_upperbound[attr_index] && value >= inv_list_lowerbound[attr_index])
        return true;
    else
        return false;
}



int
genie::table::inv_table::get_upperbound_of_list(int attr_index)
{
    if((unsigned int)attr_index < inv_list_upperbound.size())
        return inv_list_upperbound[attr_index];
    else
        return -1;
}

int
genie::table::inv_table::get_lowerbound_of_list(int attr_index)
{
    if((unsigned int)attr_index < inv_list_lowerbound.size())
        return inv_list_lowerbound[attr_index];
    else
        return -1;
}

unsigned int
genie::table::inv_table::_shift_bits_subsequence()
{
    return shift_bits_subsequence;
}

void
genie::table::inv_table::set_table_index(int attr_index)
{
    table_index = attr_index;
}
void
genie::table::inv_table::set_total_num_of_table(int num)
{
    total_num_of_table = num;
}

int
genie::table::inv_table::get_table_index() const
{
    return table_index;
}
    
int
genie::table::inv_table::get_total_num_of_table() const
{
    return total_num_of_table;
}

genie::table::inv_table::status genie::table::inv_table::build_status()
{
	return _build_status;
}

std::vector<genie::table::inv_list>*
genie::table::inv_table::inv_lists()
{
	return &_inv_lists;
}

vector<int>*
genie::table::inv_table::ck()
{
	return &_ck;
}

vector<int>*
genie::table::inv_table::inv()
{
	return &_inv;
}


unordered_map<size_t, int>*
genie::table::inv_table::inv_index_map()
{
	return &_inv_index_map;
}

vector<int>*
genie::table::inv_table::inv_pos()
{
	return &_inv_pos;
}


void
genie::table::inv_table::build(size_t max_length, bool use_load_balance)
{
    u64 table_start = getTime();
    vector<int> _inv_index;
	_ck.clear();
    _inv.clear();
	_inv_pos.clear();
    if(!use_load_balance)
    {
        max_length = (size_t)0 - (size_t)1;
    }
    unsigned int last;
	int key, dim, value;
	for (unsigned int i = 0; i < _inv_lists.size(); i++)
	{
		dim = i << _shifter;
		for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
		{
			key = dim + value - _inv_lists[i].min();
            
			vector<int>* _index;
            
            _index = _inv_lists[i].index(value);
    
            vector<int> index;
            index.clear();
            if(_index != NULL)
                index = *_index;
            if(_inv_lists.size() <= 1)//used int subsequence search
                shift_bits_subsequence = _inv_lists[i]._shift_bits_subsequence();

			if (_ck.size() <= (unsigned int) key)
			{
				last = _ck.size();
				_ck.resize(key + 1);
				_inv_index.resize(key + 1);
				for (; last < _ck.size(); ++last)
				{
					_ck[last] = _inv.size();
					_inv_index[last] = _inv_pos.size();
				}
			}
			for (unsigned int j = 0; j < index.size(); ++j)
			{
                if (j % max_length == 0)
				{
					_inv_pos.push_back(_inv.size());
				}
				_inv.push_back(index[j]);
				_ck[key] = _inv.size();
			}

		}

	}
	_inv_index.push_back(_inv_pos.size());
	_inv_pos.push_back(_inv.size());

	/* unordered map impl of inv_index */
	_inv_index_map.clear();
	for (size_t i = 0; i < _inv_lists.size(); ++i)
	{
		dim = i << _shifter;
		for (int j = _inv_lists[i].min(); j <= _inv_lists[i].max() + 1; ++j)
		{
			key = dim + j - _inv_lists[i].min();
			size_t unsigned_key = static_cast<size_t>(key);
			_inv_index_map.insert(make_pair(unsigned_key, _inv_index.at(unsigned_key)));
		}
	}

    max_inv_size = (int)_inv.size() > max_inv_size?(int)_inv.size():max_inv_size;

	_build_status = builded;
    	u64 table_end = getTime();
	cout<<"build table time = "<<getInterval(table_start, table_end)<<"ms."<<endl;
    	//Logger::log(Logger::DEBUG, "inv_index size %d:", _inv_index.size());
	//Logger::log(Logger::DEBUG, "inv_pos size %d:", _inv_pos.size());
	//Logger::log(Logger::DEBUG, "inv size %d:", _inv.size());
    	Logger::log(Logger::INFO, "inv_index size %d:", _inv_index.size());
	Logger::log(Logger::INFO, "inv_pos size %d:", _inv_pos.size());
	Logger::log(Logger::INFO, "inv size %d:", _inv.size());
}

void
genie::table::inv_table::set_min_value_sequence(int min_value)
{
    min_value_sequence = min_value;
}

int
genie::table::inv_table::get_min_value_sequence()
{
    return min_value_sequence;
}

void
genie::table::inv_table::set_max_value_sequence(int max_value)
{
    max_value_sequence = max_value;
}

int
genie::table::inv_table::get_max_value_sequence()
{
     return max_value_sequence;
}

    
void
genie::table::inv_table::set_gram_length_sequence(int gram_length)
{
    gram_length_sequence = gram_length;

}

int
genie::table::inv_table::get_gram_length_sequence()
{
    return gram_length_sequence;
}
