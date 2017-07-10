/*! \file query.cc
 *  \brief Implementation for query class
 *
 */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <iostream>

using namespace std;

#include <genie/table/inv_compr_table.h>
#include <genie/utility/Logger.h>

#include "query.h"

typedef unsigned int u32;
typedef unsigned long long u64;

GPUGenie::query::query()
{
	_ref_table = NULL;
	_attr_map.clear();
	_dim_map.clear();
	_topk = 1;
	_selectivity = -1.0f;
	_index = -1;
	_count = -1;
	is_load_balanced = false;
	use_load_balance = false;
}

GPUGenie::query::query(inv_table* ref, int index)
{
	_ref_table = ref;
	_attr_map.clear();
	_dim_map.clear();
	_topk = 1;
	_selectivity = -1.0f;
	_index = index;
	_count = -1;
	is_load_balanced = false;
	use_load_balance = false;
}

GPUGenie::query::query(inv_table& ref, int index)
{
	_ref_table = &ref;
	_attr_map.clear();
	_dim_map.clear();
	_topk = 1;
	_selectivity = -1.0f;
	_index = index;
	_count = 0;
	is_load_balanced = false;
	use_load_balance = false;
}

GPUGenie::query::dim::dim(int query, int order, int start_pos, int end_pos, float weight)
	: query(query),
	  order(order),
	  start_pos(start_pos),
	  end_pos(end_pos),
	  weight(weight) {}

GPUGenie::inv_table* GPUGenie::query::ref_table()
{
	return _ref_table;
}

void GPUGenie::query::attr(int index, int low, int up, float weight, int order)
{
	attr(index, low, up, weight, -1, order);
}

void GPUGenie::query::attr(int index, int value, float weight,
		float selectivity, int order)
{
	attr(index, value, value, weight, selectivity, order);
}

void GPUGenie::query::attr(int index, int low, int up, float weight,
		float selectivity, int order)
{
	if (index < 0 || index >= _ref_table->m_size())
		return;

	range new_attr;
	new_attr.low = low;
	new_attr.up = up;
	new_attr.weight = weight;
	new_attr.dim = index;
	new_attr.query = _index;
	new_attr.order = order;
	new_attr.low_offset = 0;
	new_attr.up_offset = 0;
	new_attr.selectivity = selectivity;

	if (_attr_map.find(index) == _attr_map.end())
	{
		//std::vector<range>* new_range_list = new std::vector<range>;
		//_attr_map[index] = new_range_list;
		_attr_map[index] = vector<range>();
	}

	_attr_map[index].push_back(new_attr);
	_count++;
}


void GPUGenie::query::clear_dim(int index)
{
	if (_attr_map.find(index) == _attr_map.end())
	{
		return;
	}
	//_count -= _attr_map[index]->size();
	//_attr_map[index]->clear();
	//free(_attr_map[index]);
	_count -= _attr_map[index].size();
	_attr_map[index].clear();
	_attr_map[index].shrink_to_fit();
	_attr_map.erase(index);
}

void GPUGenie::query::selectivity(float s)
{
	_selectivity = s;
}

float GPUGenie::query::selectivity()
{
	return _selectivity;
}

void GPUGenie::query::build_and_apply_load_balance(int max_load)
{
    this->build();

    if(max_load <= 0)
    {
        Logger::log(Logger::ALERT, "Please set a valid max_load.");
        return;
    }

    _dims.clear();

    for(auto di = _dim_map.begin() ; di != _dim_map.end(); ++di)
    {
        std::vector<dim>& dims = di->second;
        for(unsigned int i = 0; i < dims.size(); ++i)
        {
            unsigned int length = dims[i].end_pos - dims[i].start_pos;
            if((unsigned int)max_load > length)
            {
                _dims.push_back(dims[i]);
                continue;
            }
            unsigned int j = 1;
            for(; max_load*j <= length; ++j)
            {
                 dim new_dim;
                 new_dim.query = dims[i].query;
                 new_dim.order = dims[i].order;
                 new_dim.weight = dims[i].weight;
                 new_dim.start_pos = max_load*(j-1) + dims[i].start_pos;
                 new_dim.end_pos = new_dim.start_pos + max_load;
                 _dims.push_back(new_dim);
            }
            if(max_load*(j-1) != length)
            {
                 dim new_dim;
                 new_dim.query = dims[i].query;
                 new_dim.order = dims[i].order;
                 new_dim.weight = dims[i].weight;
                 new_dim.start_pos = max_load*(j-1) + dims[i].start_pos;
                 new_dim.end_pos = dims[i].end_pos;
                 _dims.push_back(new_dim);
            }


        }
    }

    this->is_load_balanced = true;
}




void GPUGenie::query::apply_adaptive_query_range()
{
	inv_table& table = *_ref_table;

	if (table.build_status() == inv_table::not_builded)
	{
		Logger::log(Logger::ALERT,
				"Please build the inverted table before applying adaptive query range.");
		return;
	}

	u32 global_threshold = _selectivity > 0 ? u32(ceil(_selectivity * table.i_size())) : 0;
	u32 local_threshold;

	for (auto di = _attr_map.begin(); di != _attr_map.end(); ++di)
	{
		std::vector<range>* ranges = &(di->second);
		int index = di->first;

		for (unsigned int i = 0; i < ranges->size(); ++i)
		{
			range& d = ranges->at(i);
			if (d.selectivity > 0)
			{
				local_threshold = ceil(d.selectivity * table.i_size());
			}
			else if (_selectivity > 0)
			{
				local_threshold = global_threshold;
			}
			else
			{
				Logger::log(Logger::ALERT, "Please set valid selectivity!");
				return;
			}

			unsigned int count = 0;
			for (int vi = d.low; vi <= d.up; ++vi)
			{
				if(!table.list_contain(index, vi))
					continue;
				count += table.get_posting_list_size(index, vi);
			}
			while (count < local_threshold)
			{
				if (d.low > 0)
				{
					d.low--;
					if(!table.list_contain(index, d.low))
						continue;
					count += table.get_posting_list_size(index, d.low);
				}
				if(!table.list_contain(index, d.up+1))
				{
					if (d.low == 0)
						break;
				}
				else
				{
					d.up++;
					count += table.get_posting_list_size(index, d.up);
				}
			}
		}
	}
}

void GPUGenie::query::topk(int k)
{
	_topk = k;
}

int GPUGenie::query::topk()
{
	return _topk;
}

void GPUGenie::query::build()
{
	int low, up;
	float weight;
	inv_table& table = *_ref_table;
	unordered_map<size_t, int>& inv_index_map = *table.inv_index_map();
    vector<int>& inv_pos = *table.inv_pos();

	for (auto di = _attr_map.begin(); di != _attr_map.end(); ++di)
	{
		int index = di->first;
		std::vector<range>& ranges = di->second;
		int d = index << _ref_table->shifter();

		if (ranges.empty())
		{
			continue;
		}

		if (_dim_map.find(index) == _dim_map.end())
		{
			//std::vector<dim>* new_list = new std::vector<dim>;
			//_dim_map[index] = new_list;
			_dim_map[index] = vector<dim>();
		}

		for (unsigned int i = 0; i < ranges.size(); ++i)
		{

			range& ran = ranges[i];
			low = ran.low;
			up = ran.up;
			weight = ran.weight;

            if(low > up || low > table.get_upperbound_of_list(index) || up < table.get_lowerbound_of_list(index))
			{
				continue;
			}

			dim new_dim;
			new_dim.weight = weight;
			new_dim.query = _index;
			new_dim.order = ran.order;

            low = low < table.get_lowerbound_of_list(index)?table.get_lowerbound_of_list(index):low;

            up = up > table.get_upperbound_of_list(index)?table.get_upperbound_of_list(index):up;


            int _min, _max;

            _min = d + low - table.get_lowerbound_of_list(index);
            _max = d + up - table.get_lowerbound_of_list(index);

            new_dim.start_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_min))->second];
            new_dim.end_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_max+1))->second];

			_dim_map[index].push_back(new_dim);
		}
	}
}

void GPUGenie::query::build(vector<dim> &dims)
{
	int low, up, order, start_pos, end_pos;
	float weight;
	unordered_map<size_t, int> &inv_index_map = *_ref_table->inv_index_map();
	vector<int> &inv_pos = *_ref_table->inv_pos();

	for (auto di = _attr_map.begin(); di != _attr_map.end(); ++di)
	{
		int index = di->first;
		vector<range> &ranges = di->second;
		int d = index << _ref_table->shifter();

		if (ranges.empty())
			continue;

		for (size_t i = 0; i < ranges.size(); ++i)
		{
			range &ran = ranges[i];
			low = ran.low;
			up = ran.up;
			weight = ran.weight;
			order = ran.order;

			if (low > up || low > _ref_table->get_upperbound_of_list(index) || up < _ref_table->get_lowerbound_of_list(index))
				continue;

			low = low < _ref_table->get_lowerbound_of_list(index) ? _ref_table->get_lowerbound_of_list(index) : low;
			up = up > _ref_table->get_upperbound_of_list(index) ? _ref_table->get_upperbound_of_list(index) : up;

			int _min, _max;

			_min = d + low - _ref_table->get_lowerbound_of_list(index);
			_max = d + up - _ref_table->get_lowerbound_of_list(index);

			start_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_min))->second];
			end_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_max+1))->second];

			dims.emplace_back(_index, order, start_pos, end_pos, weight);
		}
	}
}

void GPUGenie::query::build_sequence()
{
	int low, up;
	float weight;
	inv_table& table = *_ref_table;
    unordered_map<size_t, int>& inv_index_map = *table.inv_index_map();
    vector<int>& inv_pos = *table.inv_pos();

    //unordered_map<int, int> _distinct;

	for (auto di = _attr_map.begin(); di != _attr_map.end(); ++di)
	{
        //_distinct.clear();
		int index = di->first;
        unordered_map<int, int>& _distinct = *table.get_distinct_map(index);
        if(_distinct.size() <= 0) continue;

        unordered_map<int, int>::iterator itt = _distinct.begin();

		std::vector<range>& ranges = di->second;
		int d = index << _ref_table->shifter();


		if (ranges.empty())
		{
			continue;
		}

		if (_dim_map.find(index) == _dim_map.end())
		{
			//std::vector<dim>* new_list = new std::vector<dim>;
			//_dim_map[index] = new_list;
			_dim_map[index] = vector<dim>();
		}
        
		for (unsigned int i = 0; i < ranges.size(); ++i)
		{

			range& ran = ranges[i];
			low = ran.low;
			up = ran.up;
			weight = ran.weight;
            //Here low must equal up
            //vector<int>::iterator it = std::find(distinct_value.begin(), distinct_value.end(), low);
            unordered_map<int, int>::iterator it = _distinct.find(low);
            if(it != _distinct.end())
            {
                 low = it->second;
                 up = low;
            }
            else
                continue;


            if(low > up || low > table.get_upperbound_of_list(index) || up < table.get_lowerbound_of_list(index))
			{
				continue;
			}

			dim new_dim;
			new_dim.weight = weight;
			new_dim.query = _index;

            if(low < table.get_lowerbound_of_list(index))
                continue;

            if(up > table.get_upperbound_of_list(index))
                continue;

            int _min, _max;

            _min = d + low - table.get_lowerbound_of_list(index);
            _max = d + up - table.get_lowerbound_of_list(index);

            new_dim.start_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_min))->second];
            new_dim.end_pos = inv_pos[inv_index_map.find(static_cast<size_t>(_max+1))->second];

			_dim_map[index].push_back(new_dim);
		}


	}
}


#ifdef GENIE_COMPR 
void GPUGenie::query::build_compressed(int max_load)
{
    inv_compr_table& table = *dynamic_cast<inv_compr_table*>(_ref_table);
    unordered_map<size_t, int>& inv_index_map = *table.inv_index_map();
    vector<int>& inv_pos = *table.inv_pos();

    for (std::map<int, std::vector<range>>::iterator di = _attr_map.begin();
            di != _attr_map.end(); ++di)
    {
        int index = di->first;
        std::vector<range> &ranges = di->second;
        int d = index << _ref_table->shifter();

        if (ranges.empty())
            continue;

        if (_dim_map.find(index) == _dim_map.end())
            _dim_map[index] = std::vector<dim>();

        for (unsigned int i = 0; i < ranges.size(); ++i)
        {

            range& ran = ranges[i];
            int low = ran.low;
            int up = ran.up;

            if(low > up || low > table.get_upperbound_of_list(index) || up < table.get_lowerbound_of_list(index))
                continue;

            low = std::max(low, table.get_lowerbound_of_list(index));
            up = std::min(up, table.get_upperbound_of_list(index));

            int _min, _max;
            _min = d + low - table.get_lowerbound_of_list(index);
            _max = d + up - table.get_lowerbound_of_list(index);

            // Index to inv_pos array of the first valid inverted list
            int beginInvListIndex = inv_index_map.find(static_cast<size_t>(_min))->second; 
            // Index to inv_pos array after the last valid inverted list
            int endInvListIndex = inv_index_map.find(static_cast<size_t>(_max+1))->second; 

            assert (endInvListIndex > beginInvListIndex);

            if (max_load <= 0){
                // Check that we are definitely not using multirange, i.e. check there is only one inv list in the
                // compiled query, in case max_load is not specified
                assert(beginInvListIndex + 1 == endInvListIndex);
            } else {
                // Check that all the inverted lists in the compiled query will be smaller than max load
                for (int ipos = beginInvListIndex; ipos < endInvListIndex; ipos++){
                    assert(inv_pos[ipos+1] - inv_pos[ipos+1] <= max_load);
                }
            }

            // Generate compiled queries
            std::vector<dim> &compiledQueriesVec = _dim_map[index];
            for (int ipos = beginInvListIndex; ipos < endInvListIndex; ipos++){
                dim new_dim;
                new_dim.weight = ran.weight;
                new_dim.query = _index;
                new_dim.order = ran.order;
                new_dim.start_pos = inv_pos[ipos];
                new_dim.end_pos = inv_pos[ipos+1];

                compiledQueriesVec.push_back(new_dim);
            }
        }

    }
}
#endif

int GPUGenie::query::dump(vector<dim>& vout)
{
    if (is_load_balanced)
    {
        for(unsigned int i = 0; i < _dims.size(); ++i)
        {
            vout.push_back(_dims[i]);
        }
        return _dims.size();
    }
    int count = 0;
    for (std::map<int, std::vector<dim>>::iterator di = _dim_map.begin();
            di != _dim_map.end(); ++di)
    {
        std::vector<dim> &ranges = di->second;
        count += ranges.size();

            //vector<int>& _inv_ = *_ref_table->inv();
        for (unsigned int i = 0; i < ranges.size(); ++i)
        {
            vout.push_back(ranges[i]);
        }
    }
    return count;
}


int GPUGenie::query::dump(vector<range>& ranges)
{
    int count = 0;
    ranges.clear();
    for (std::map<int, std::vector<range>>::iterator di = _attr_map.begin();
            di != _attr_map.end(); ++di)
    {
        std::vector<range> &rangesInAttr = di->second;
        count += rangesInAttr.size();

        for (range &r : rangesInAttr)
            ranges.push_back(r);
    }
	return count;
}

int GPUGenie::query::index()
{
	return _index;
}

int GPUGenie::query::count_ranges()
{
	return _count;
}

void GPUGenie::query::print(int limit)
{
	Logger::log(Logger::DEBUG, "This query has %d dimensions.",
			_attr_map.size());
	int count = limit;
	for (auto di = _attr_map.begin(); di != _attr_map.end(); ++di)
	{
		std::vector<range>& ranges = di->second;
		for (unsigned int i = 0; i < ranges.size(); ++i)
		{
			if (count == 0)
				return;
			if (count > 0)
				count--;
			range& d = ranges[i];
			Logger::log(Logger::DEBUG,
					"dim %d from query %d: low %d(offset %d), up %d(offset %d), weight %.2f, selectivity %.2f.",
					d.dim, d.query, d.low, d.low_offset, d.up, d.up_offset,
					d.weight, d.selectivity);
		}
	}
}


