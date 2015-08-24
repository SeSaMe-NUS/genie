#include "query.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int u32;
typedef unsigned long long u64;

GaLG::query::query(inv_table* ref)
{
  _ref_table = ref;
  _attr.clear();
  _attr.resize(_ref_table->m_size());
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      _attr[i].low = 0;
      _attr[i].up = -1;
      _attr[i].weight = 0;

      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
  _topk = 1;
  _selectivity = -1.0f;
}

GaLG::query::query(inv_table& ref)
{
  _ref_table = &ref;
  _attr.clear();
  _attr.resize(_ref_table->m_size());
  _dims.clear();
  _dims.resize(_ref_table->m_size());
  int i;
  for (i = 0; i < _dims.size(); i++)
    {
      _attr[i].low = 0;
      _attr[i].up = -1;
      _attr[i].weight = 0;

      /* Mark always not match */
      _dims[i].low = 0;
      _dims[i].up = -1;
      _dims[i].weight = 0;
    }
  _topk = 1;
  _selectivity = -1.0f;
}

GaLG::inv_table*
GaLG::query::ref_table()
{
  return _ref_table;
}

void
GaLG::query::attr(int index, int low, int up, float weight)
{
  if (index < 0 || index >= _dims.size())
    return;

  _attr[index].low = low;
  _attr[index].up = up;
  _attr[index].weight = weight;
}

inline u64
GaLG::query::pack_dim_and_count(u32 dim, u64 count)
{
	u64 mask = u64((1u << 16) - 1u);
	return (u64(count) << 16) + (mask & u64(dim));
}
inline u32
GaLG::query::unpack_dim(u64 packed_data)
{
	u64 mask = u64((1u << 16) - 1u);
	return packed_data & mask;
}
inline u64
GaLG::query::unpack_count(u64 packed_data)
{
	return packed_data >> 16;
}

void
GaLG::query::clear_dim(int index)
{
	_attr[index].low = 0;
	_attr[index].up = -1;
	_attr[index].weight = 0.0f;
}

void
GaLG::query::split_hot_dims(GaLG::query& hot_dims_query, int num)
{
	std::vector<inv_list>& inv_lists = *((*_ref_table).inv_lists());
	std::vector<u64> counts;
	counts.resize(inv_lists.size());
	u64 count_items_on_dim;

	for(int di = 0; di < inv_lists.size(); ++di)
	{
		if(_attr[di].up == -1) continue;

		count_items_on_dim = 0ull;
		for(int ni = _attr[di].low; ni <= _attr[di].up; ++ni)
		{
			if(!inv_lists[di].contains(ni)) continue;

			std::vector<int> index_list = *(inv_lists[di].index(ni));
			count_items_on_dim += index_list.size();
		}

		counts.push_back(pack_dim_and_count(u32(di), count_items_on_dim));
	}

	std::make_heap(counts.begin(), counts.end());
	std::vector<int> hot_index;


	//extract top [num] hot dim index
	for(int i = 0; i < num && i < _ref_table->m_size(); ++i)
	{
		hot_index.push_back(unpack_dim(counts.front()));
		std::pop_heap(counts.begin(), counts.end());
		counts.pop_back();
	}

	//cut hot dims from self to the given hot_dims_query
	for(int i = 0; i < hot_index.size(); ++i)
	{
		int index = hot_index[i];
		if (index < 0 || index >= _attr.size())
			continue;
		hot_dims_query.attr(index, _attr[index].low, _attr[index].up, _attr[index].weight);
		clear_dim(index);
	}
}

void
GaLG::query::selectivity(float s)
{
	if(s > 0) _selectivity = s;
	else printf("Please choose a selectivity greater than 0.\n");
}

float
GaLG::query::selectivity()
{
	return _selectivity;
}

void
GaLG::query::apply_adaptive_query_range()
{
	inv_table& table = *_ref_table;
	std::vector<inv_list>& lists = *table.inv_lists();

	if(table.build_status() == inv_table::not_builded)
	{
		printf("Please build the inverted table before applying adaptive query range.\n");
		return;
	}

	if(_selectivity <= 0.0f)
	{
		printf("Please set a valid selectivity.\n");
		return;
	}

	u32 threshold = u32(_selectivity * table.i_size());
	if(_selectivity * table.i_size() - threshold > 0)
		threshold += 1;

	for(int di = 0; di < _attr.size(); ++di)
	{
		dim& d = _attr[di];
		if(d.up == -1) continue;
		int count = 0;
		for(int vi = d.low; vi <= d.up; ++vi)
		{
			if(!lists[di].contains(vi)) continue;
			count += lists[di].index(vi)->size();
		}
		while(count < threshold)
		{
			if(d.low > 0)
			{
				d.low --;
				if(!lists[di].contains(d.low)) continue;
				count += lists[di].index(d.low)->size();
			}

			if(!lists[di].contains(d.up + 1))
			{
				if(d.low == 0) break;
			} else
			{
				d.up++;
				count += lists[di].index(d.up)->size();
			}
		}
	}

}

void
GaLG::query::topk(int k)
{
  _topk = k;
}

int
GaLG::query::topk()
{
  return _topk;
}

void
GaLG::query::build()
{
  int index, low, up;
  float weight;
  for (index = 0; index < _attr.size(); index++)
    {
      low = _attr[index].low;
      up = _attr[index].up;
      weight = _attr[index].weight;

      if (low > up)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;;
        }

      int d = index << _ref_table->shifter();
      vector<inv_list>& inv_lists = *_ref_table->inv_lists();
      inv_list& inv = inv_lists[index];
      _dims[index].weight = weight;

      if (low < inv.min())
        {
          low = inv.min();
        }

      if (low > inv.max())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      if (up > inv.max())
        {
          up = inv.max();
        }

      if (up < inv.min())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      _dims[index].low = d + low - inv.min();
      _dims[index].up = d + up - inv.min();
    }
}

void
GaLG::query::build_compressed()
{
  int index, low, up;
  float weight;
  map<int, int>::iterator lower_bound;

  for (index = 0; index < _attr.size(); index++)
    {
      low = _attr[index].low;
      up = _attr[index].up;
      weight = _attr[index].weight;

      if (low > up)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;;
        }

      int d = index << _ref_table->shifter();
      vector<inv_list>& inv_lists = *_ref_table->inv_lists();
      inv_list& inv = inv_lists[index];
      _dims[index].weight = weight;

      if (low < inv.min())
        {
          low = inv.min();
        }

      if (low > inv.max())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      if (up > inv.max())
        {
          up = inv.max();
        }

      if (up < inv.min())
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }

      lower_bound = _ref_table->ck_map()->lower_bound(d + low - inv.min());
      _dims[index].low = lower_bound->second;

      lower_bound = _ref_table->ck_map()->lower_bound(d + up - inv.min());
      if (lower_bound->first - d + inv.min() != up)
        {
          lower_bound--;
        }
      _dims[index].up = lower_bound->second;

      if (_dims[index].up < _dims[index].low)
        {
          /* Mark always not match */
          _dims[index].low = 0;
          _dims[index].up = -1;
          continue;
        }
    }
}

void
GaLG::query::dump(vector<dim>& vout)
{
  int i;
  for (i = 0; i < _dims.size(); i++)
    vout.push_back(_dims[i]);
}

void
GaLG::query::print()
{
	printf("This query has %d dimensions.\n", _attr.size());
	for(int i = 0; i < _attr.size(); ++i)
	{
		printf("dim %d: low %d, up %d, weight %.2f.\n", i, _attr[i].low, _attr[i].up, _attr[i].weight);
	}
}
