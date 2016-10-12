/*! \file inv_table.cu
 *  \brief Implementation of class inv_table 
 *  declared in file inv_table.h
 */


#include <stdio.h>
#include <fstream>
#include <exception>
#include <iostream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp>
#include "Logger.h"
#include "genie_errors.h"

#include "inv_table.h"
#include "Timing.h"
using namespace std;
using namespace GPUGenie;

int*  inv_table::d_inv_p = NULL;


bool GPUGenie::inv_table::cpy_data_to_gpu()
{
	try{
        if(d_inv_p == NULL)
        {
            cudaCheckErrors(cudaMalloc(&d_inv_p, sizeof(int) * _inv.size()));
        }
            u64 t=getTime();
		cudaCheckErrors(cudaMemcpy(d_inv_p, &_inv[0], sizeof(int) * _inv.size(),cudaMemcpyHostToDevice));
        	u64 tt=getTime();
        	cout<<"The inverted list(all data) transfer time = "<<getInterval(t,tt)<<"ms"<<endl;
	} catch(std::bad_alloc &e){
		throw(GPUGenie::gpu_bad_alloc(e.what()));
	}

	return true;
}

void GPUGenie::inv_table::clear()
{
	_build_status = not_builded;
	_inv_lists.clear();
	_ck.clear();
	_inv.clear();
}

GPUGenie::inv_table::~inv_table()
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
void GPUGenie::inv_table::clear_gpu_mem()
{
	if (d_inv_p == NULL)
		return;


    cout << "cudaFreeTime: " ;
	u64 t1 = getTime();
    cudaCheckErrors(cudaFree(d_inv_p));
    u64 t2 = getTime();
    cout << getInterval(t1, t2) << " ms."<< endl;

}

bool GPUGenie::inv_table::empty()
{
	return _size == -1;
}

int GPUGenie::inv_table::m_size()
{
    return _dim_size;
	//return _inv_lists.size();
}

int GPUGenie::inv_table::i_size()
{
	return _size <= -1 ? 0 : _size;
}

int GPUGenie::inv_table::shifter()
{
	return _shifter;
}

unordered_map<int, int>* GPUGenie::inv_table::get_distinct_map(int dim)
{
    return &_distinct_map[dim];
}

void GPUGenie::inv_table::append(inv_list& inv)
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

void GPUGenie::inv_table::append_sequence(inv_list& inv)
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

void GPUGenie::inv_table::append(inv_list* inv)
{
	if (inv != NULL)
	{
		append(*inv);
	}
}

int
GPUGenie::inv_table::get_posting_list_size(int attr_index, int value)
{
    if((unsigned int)attr_index<posting_list_size.size() && value>=inv_list_lowerbound[attr_index] && value<=inv_list_upperbound[attr_index])
        return posting_list_size[attr_index][value-inv_list_lowerbound[attr_index]];
    else
        return 0;
}

bool
GPUGenie::inv_table::list_contain(int attr_index, int value)
{
    if(value <= inv_list_upperbound[attr_index] && value >= inv_list_lowerbound[attr_index])
        return true;
    else
        return false;
}



int
GPUGenie::inv_table::get_upperbound_of_list(int index)
{
    if((unsigned int)index < inv_list_upperbound.size())
        return inv_list_upperbound[index];
    else
        return -1;
}

int
GPUGenie::inv_table::get_lowerbound_of_list(int index)
{
    if((unsigned int)index < inv_list_lowerbound.size())
        return inv_list_lowerbound[index];
    else
        return -1;
}

unsigned int
GPUGenie::inv_table::_shift_bits_subsequence()
{
    return shift_bits_subsequence;
}

void
GPUGenie::inv_table::set_table_index(int index)
{
    table_index = index;
}
void
GPUGenie::inv_table::set_total_num_of_table(int num)
{
    total_num_of_table = num;
}

int
GPUGenie::inv_table::get_table_index()
{
    return table_index;
}
    
int
GPUGenie::inv_table::get_total_num_of_table()
{
    return total_num_of_table;
}

GPUGenie::inv_table::status GPUGenie::inv_table::build_status()
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


void
GPUGenie::inv_table::build(u64 max_length, bool use_load_balance)
{
    u64 table_start = getTime();
	_ck.clear(), _inv.clear();
	_inv_index.clear();
	_inv_pos.clear();
    if(!use_load_balance)
    {
        max_length = (u64)0 - (u64)1;
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



bool
GPUGenie::inv_table::write_to_file(ofstream& ofs)
{
    if(_build_status == not_builded)
        return false;

    ofs.write((char*)&table_index, sizeof(int));
    ofs.write((char*)&total_num_of_table, sizeof(int));
    ofs.write((char*)&_shifter, sizeof(int));
    ofs.write((char*)&_size, sizeof(int));
    ofs.write((char*)&_dim_size, sizeof(int));
    ofs.write((char*)&shift_bits_subsequence, sizeof(unsigned int));
    ofs.write((char*)&min_value_sequence, sizeof(int));
    ofs.write((char*)&max_value_sequence, sizeof(int));
    ofs.write((char*)&gram_length_sequence, sizeof(int));
    ofs.write((char*)&shift_bits_sequence, sizeof(int));
    int temp_status = _build_status;
    ofs.write((char*)&temp_status, sizeof(int));

    unsigned int _ck_size = _ck.size();
    unsigned int _inv_size = _inv.size();
    unsigned int _inv_index_size = _inv_index.size();
    unsigned int _inv_pos_size = _inv_pos.size();
    
    ofs.write((char*)&_ck_size, sizeof(unsigned int));
    ofs.write((char*)&_inv_size, sizeof(unsigned int));
    ofs.write((char*)&_inv_index_size, sizeof(unsigned int));
    ofs.write((char*)&_inv_pos_size, sizeof(unsigned int));

    ofs.write((char*)&_ck[0], _ck_size*sizeof(int));
    ofs.write((char*)&_inv[0], _inv_size*sizeof(int));
    ofs.write((char*)&_inv_index[0],_inv_index_size*sizeof(int));
    ofs.write((char*)&_inv_pos[0], _inv_pos_size*sizeof(int));

    unsigned int _list_upperbound_size = inv_list_upperbound.size();
    unsigned int _list_lowerbound_size = inv_list_lowerbound.size();

    ofs.write((char*)&_list_upperbound_size, sizeof(unsigned int));
    ofs.write((char*)&_list_lowerbound_size, sizeof(unsigned int));

    ofs.write((char*)&inv_list_upperbound[0], _list_upperbound_size*sizeof(int));
    ofs.write((char*)&inv_list_lowerbound[0], _list_lowerbound_size*sizeof(int));

    //write posting list size
    unsigned int num_of_attr = posting_list_size.size();
    ofs.write((char*)&num_of_attr, sizeof(unsigned int));
    for(unsigned int i=0 ; i<num_of_attr ; ++i)
    {
         unsigned int value_range_size = posting_list_size[i].size();
         ofs.write((char*)&value_range_size, sizeof(unsigned int));
         ofs.write((char*)&posting_list_size[i][0], value_range_size*sizeof(int));
    }
    //write distinct values
    
    if(gram_length_sequence != 1)
    {
        unsigned int num_of_entry = _distinct_map.size();
        ofs.write((char*)&num_of_entry, sizeof(unsigned int));
        boost::archive::binary_oarchive oa(ofs);
        for(unsigned int i=0 ; i<num_of_entry ; ++i)
        {
            oa<<_distinct_map[i];
        }
    }



    if(table_index == total_num_of_table - 1)
        ofs.close();
    return true;
}


bool
GPUGenie::inv_table::read_from_file(ifstream& ifs)
{
    
    ifs.read((char*)&table_index, sizeof(int));
    ifs.read((char*)&total_num_of_table, sizeof(int));
    ifs.read((char*)&_shifter, sizeof(int));
    ifs.read((char*)&_size, sizeof(int));
    ifs.read((char*)&_dim_size, sizeof(int));
    ifs.read((char*)&shift_bits_subsequence, sizeof(unsigned int));
    ifs.read((char*)&min_value_sequence, sizeof(int));
    ifs.read((char*)&max_value_sequence, sizeof(int));
    ifs.read((char*)&gram_length_sequence, sizeof(int));
    ifs.read((char*)&shift_bits_sequence, sizeof(int));
    int temp_status;
    ifs.read((char*)&temp_status, sizeof(int));
    _build_status = static_cast<status>(temp_status);


    unsigned int _ck_size;
    unsigned int _inv_size;
    unsigned int _inv_index_size;
    unsigned int _inv_pos_size;

    ifs.read((char*)&_ck_size, sizeof(unsigned int));
    ifs.read((char*)&_inv_size, sizeof(unsigned int));
    ifs.read((char*)&_inv_index_size, sizeof(unsigned int));
    ifs.read((char*)&_inv_pos_size, sizeof(unsigned int));

    _ck.resize(_ck_size);
    _inv.resize(_inv_size);
    _inv_index.resize(_inv_index_size);
    _inv_pos.resize(_inv_pos_size);

    ifs.read((char*)&_ck[0], _ck_size*sizeof(int));
    ifs.read((char*)&_inv[0], _inv_size*sizeof(int));
    ifs.read((char*)&_inv_index[0],_inv_index_size*sizeof(int));
    ifs.read((char*)&_inv_pos[0], _inv_pos_size*sizeof(int));
    
    unsigned int _list_upperbound_size;
    unsigned int _list_lowerbound_size;

    ifs.read((char*)&_list_upperbound_size, sizeof(unsigned int));
    ifs.read((char*)&_list_lowerbound_size, sizeof(unsigned int));

    inv_list_upperbound.resize(_list_upperbound_size);
    inv_list_lowerbound.resize(_list_lowerbound_size);
    ifs.read((char*)&inv_list_upperbound[0], _list_upperbound_size*sizeof(int));
    ifs.read((char*)&inv_list_lowerbound[0], _list_lowerbound_size*sizeof(int));

    unsigned int num_of_attr;
    ifs.read((char*)&num_of_attr, sizeof(unsigned int));
    posting_list_size.resize(num_of_attr);
    for(unsigned int i=0 ; i<num_of_attr ; ++i)
    {
         unsigned int value_range_size;
         ifs.read((char*)&value_range_size, sizeof(unsigned int));
         posting_list_size[i].resize(value_range_size);
         ifs.read((char*)&posting_list_size[i][0], value_range_size*sizeof(int));
    }

    //read distinct values
    if(gram_length_sequence != 1)
    {
        unsigned int num_of_entry;
        ifs.read((char*)&num_of_entry, sizeof(unsigned int));
        _distinct_map.resize(num_of_entry);
        boost::archive::binary_iarchive ia(ifs);
        for(unsigned int i=0 ; i<num_of_entry ; ++i)
        {
            ia>>_distinct_map[i];
        }
    }
    if(table_index == total_num_of_table-1)
        ifs.close();
    
    return true;
}


bool
GPUGenie::inv_table::write(const char* filename, inv_table*& table)
{
    
    int _table_index = table[0].get_table_index();
    if(_table_index != 0)
        return false;
    
    ofstream _ofs(filename, ios::binary|ios::trunc|ios::out);
    if(!_ofs.is_open())
        return false;
    int _total_num_of_table = table[0].get_total_num_of_table();
    bool success;
    for(int i=0; i<_total_num_of_table; ++i)
    {
        success = table[i].write_to_file(_ofs);
    }
    

    return !_ofs.is_open() && success;

    
}

bool
GPUGenie::inv_table::read(const char* filename, inv_table*& table)
{
    ifstream ifs(filename, ios::binary|ios::in);
    if(!ifs.is_open())
        return false;
    
    int _table_index, _total_num_of_table;
    ifs.read((char*)&_table_index, sizeof(int));
    ifs.read((char*)&_total_num_of_table, sizeof(int));
    ifs.close();
    if(_table_index!=0 || _total_num_of_table<1)
        return false;
    
    table = new inv_table[_total_num_of_table];
    ifstream _ifs(filename, ios::binary|ios::in);
    
    bool success;
    for(int i=0 ; i<_total_num_of_table ; ++i)
    {
         success = table[i].read_from_file(_ifs);
    }
    return !_ifs.is_open() && success;
}

void
GPUGenie::inv_table::set_min_value_sequence(int min_value)
{
    min_value_sequence = min_value;
}

int
GPUGenie::inv_table::get_min_value_sequence()
{
    return min_value_sequence;
}

void
GPUGenie::inv_table::set_max_value_sequence(int max_value)
{
    max_value_sequence = max_value;
}

int
GPUGenie::inv_table::get_max_value_sequence()
{
     return max_value_sequence;
}

    
void
GPUGenie::inv_table::set_gram_length_sequence(int gram_length)
{
    gram_length_sequence = gram_length;

}

int
GPUGenie::inv_table::get_gram_length_sequence()
{
    return gram_length_sequence;
}
