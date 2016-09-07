/*! \file inv_list.cc
 *  \brief Implementation for inv_list class
 */

#include "inv_list.h"

#include <cstdlib>
#include <iostream>

int cv(string& s)
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

int GPUGenie::inv_list::min()
{
	return _bound.first;
}

int GPUGenie::inv_list::max()
{
	return _bound.second;
}

int GPUGenie::inv_list::size()
{
	return _size;
}
void GPUGenie::inv_list::invert_bijectMap(vector<vector<int> > & vin)
{
	_size = vin.size();
	if (vin.empty())
		return;

	_bound.first = vin[0][0], _bound.second = vin[0][0], _inv.clear();

	unsigned int i, j;
	for (i = 0; i < vin.size(); i++)
	{
		for (j = 0; j < vin[i].size(); ++j)
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
		for (j = 0; j < vin[i].size(); ++j)
		{
			_inv[vin[i][j] - _bound.first].push_back(i);
		}
	}
    shift_bits_subsequence = 0;
	return;
}


void GPUGenie::inv_list::invert_sequence(vector<vector<int> > & vin, int & shift_bits, vector<int> & respective_id)
{
	_size = vin.size();
	if (vin.empty())
    {
        _bound.first = 0;
        _bound.second = 0;

		return;
    }


	unsigned int i, j;
	_bound.first = 0, _bound.second = 0, _inv.clear();

    vector<vector<int> > vin_after_shift;
    shift_bits = 6;
    unordered_map<int, int> _map;
    for(i = 0; i < vin.size(); ++i)
    {
        vector<int> line;
        for(j = 0; j < vin[i].size(); ++j)
        {
            int temp_value;
            unordered_map<int, int>::iterator result = _map.find(vin[i][j]);
            if(result == _map.end())
            {
                temp_value = vin[i][j]<<shift_bits;
                _map.insert({vin[i][j], 0});
            }
            else
            {
                if(result->second < 63)
                    result->second += 1;
                temp_value = (result->first<<shift_bits) + result->second;
            }

            unordered_map<int, int>::iterator it = _distinct.find(temp_value);
            if(it != _distinct.end())
                temp_value = it->second;
            else
            {
                _distinct.insert({temp_value, _distinct.size()});
                //distinct_value_sequence.push_back(temp_value);
                temp_value = _distinct.size() - 1;
            }

            line.push_back(temp_value);
        }
        vin_after_shift.push_back(line);
        _map.clear();
    }
    _bound.second = _distinct.size() - 1;

	unsigned int gap = _bound.second - _bound.first + 1;
    //cout<<"Gap = "<<gap<<endl;
	_inv.resize(gap);
	for (i = 0; i < gap; i++)
		_inv[i].clear();

    for (i = 0; i < vin_after_shift.size(); ++i)
	{
		for (j = 0; j < vin_after_shift[i].size(); ++j)
		{
			_inv[vin_after_shift[i][j] - _bound.first].push_back(respective_id[i]);
		}
	}
    //cout<<"inv_list finished!"<<endl;
	return;
}


void GPUGenie::inv_list::invert_bijectMap(int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num)
{
	_size = row_num;
	if (item_num == 0)
		return;
	unsigned int i, j;
	for (i = 0; i < item_num; ++i)
	{
		if (_bound.first > data[i])
			_bound.first = data[i];
		if (_bound.second < data[i])
			_bound.second = data[i];
	}
	unsigned int gap = _bound.second - _bound.first + 1;
	_inv.resize(gap);
	for (i = 0; i < gap; ++i)
		_inv[i].clear();
	for (i = 0; i < row_num - 1; ++i)
		for (j = index[i] - index[0]; j < index[i + 1] - index[0]; ++j)
			_inv[data[j] - _bound.first].push_back(i);

	for (i = index[row_num - 1]; i < item_num; ++i)
		_inv[data[i] - _bound.first].push_back(row_num - 1);

    shift_bits_subsequence = 0;
	return;
}

void GPUGenie::inv_list::invert_subsequence(vector<vector<int> > & vin)
{
	_size = vin.size();
	if (vin.empty())
		return;

	_bound.first = vin[0][0], _bound.second = vin[0][0], _inv.clear();

	unsigned int i, j, max_offset_subsequence=0;
	for (i = 0; i < vin.size(); i++)
	{
        if(vin[i].size() > max_offset_subsequence)
            max_offset_subsequence = vin[i].size();
		for (j = 0; j < vin[i].size(); ++j)
		{
			if (_bound.first > vin[i][j])
				_bound.first = vin[i][j];
			if (_bound.second < vin[i][j])
				_bound.second = vin[i][j];
		}
	}
	unsigned int gap = _bound.second - _bound.first + 1;
	_inv.resize(gap);

    shift_bits_subsequence = 0;
    for(i  = 1 ; i < max_offset_subsequence ; i *= 2)
        shift_bits_subsequence++;

	for (i = 0; i < gap; i++)
		_inv[i].clear();

    unsigned int rowID_offset;
	for (i = 0; i < vin.size(); i++)
	{
		for (j = 0; j < vin[i].size(); ++j)
        {
            rowID_offset = (i<<shift_bits_subsequence) + j;
			_inv[vin[i][j] - _bound.first].push_back(rowID_offset);
		}
	}
	return;
}

void GPUGenie::inv_list::invert_subsequence(int *data, unsigned int item_num, unsigned int * index, unsigned int row_num)
{
	_size = row_num;
	if (item_num == 0)
		return;
	unsigned int i, j, max_offset_subsequence=0;
	unsigned int length;
	_bound.first = data[0], _bound.second = data[0], _inv.clear();
    for( i = 1; i < row_num ;++i )
    {
        length = index[i] - index[i-1];
        if(length > max_offset_subsequence)
            max_offset_subsequence = length;
    }
    length = item_num - index[row_num - 1];
    if(length > max_offset_subsequence)
        max_offset_subsequence = length;

    for (i = 0; i < item_num; ++i)
	{
		if (_bound.first > data[i])
			_bound.first = data[i];
		if (_bound.second < data[i])
			_bound.second = data[i];
	}
	unsigned int gap = _bound.second - _bound.first + 1;
	_inv.resize(gap);

    shift_bits_subsequence = 0;
    for(i  = 1 ; i < max_offset_subsequence ; i *= 2)
        shift_bits_subsequence++;

    for (i = 0; i < gap; ++i)
		_inv[i].clear();

    unsigned int rowID_offset;
    int offset;
    for (i = 0; i < row_num - 1; ++i)
    {
        offset = 0;
		for (j = index[i] - index[0]; j < index[i + 1] - index[0]; ++j)
        {
            rowID_offset = (i<<shift_bits_subsequence) + offset;
            _inv[data[j] - _bound.first].push_back(rowID_offset);
            offset++;
        }
    }
    offset = 0;
    int rowID = (row_num - 1)<<shift_bits_subsequence;
	for (i = index[row_num - 1] - index[0]; i < item_num; ++i)
    {
        rowID_offset = rowID + offset;
		_inv[data[i] - _bound.first].push_back(rowID_offset);
        offset++;
    }
	return;
}

unsigned int GPUGenie::inv_list::_shift_bits_subsequence()
{
     return shift_bits_subsequence;
}

void GPUGenie::inv_list::invert(vector<int>& vin)
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

void GPUGenie::inv_list::invert(vector<int>* vin)
{
	invert(*vin);
}

void GPUGenie::inv_list::invert(vector<string>& vin)
{
	invert(vin, &cv);
}

void GPUGenie::inv_list::invert(vector<string>* vin)
{
	invert(*vin);
}

void GPUGenie::inv_list::invert(vector<string>& vin,
		int (*stoi)(string&))
{
	_size = vin.size();
	if (vin.empty())
		return;

	_bound.first = stoi(vin[0]);
	_bound.second = stoi(vin[0]);
	_inv.clear();

	unsigned int i;
	for (i = 0; i < vin.size(); i++)
	{
		if (_bound.first > stoi(vin[i]))
			_bound.first = stoi(vin[i]);
		if (_bound.second < stoi(vin[i]))
			_bound.second = stoi(vin[i]);
	}

	unsigned int gap = _bound.second - _bound.first + 1;
	_inv.resize(gap);
	for (i = 0; i < gap; i++)
		_inv[i].clear();

	for (i = 0; i < vin.size(); i++)
		_inv[stoi(vin[i]) - _bound.first].push_back(i);
	return;
}

void GPUGenie::inv_list::invert(vector<string>* vin,
		int (*stoi)(string&))
{
	invert(*vin, stoi);
}

bool GPUGenie::inv_list::contains(int value)
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

int
GPUGenie::inv_list::value_range()
{
    return _inv.size();
}
