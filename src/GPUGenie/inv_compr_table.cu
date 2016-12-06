#include "inv_compr_table.h"

void
GPUGenie::inv_compr_table::build(u64 max_length, bool use_load_balance)
{
    inv_table::build(max_length, use_load_balance);


    //  u64 table_start = getTime();
    // _ck.clear();
    // _inv.clear();
    // _inv_index.clear();
    // _inv_pos.clear();
    // if(!use_load_balance)
    // {
    //     max_length = (u64)0 - (u64)1;
    // }
    // unsigned int last;
    // int key, dim, value;
    // for (unsigned int i = 0; i < _inv_lists.size(); i++)
    // {
    //     dim = i << _shifter;
    //     for (value = _inv_lists[i].min(); value <= _inv_lists[i].max(); value++)
    //     {
    //         key = dim + value - _inv_lists[i].min();
            
    //         vector<int>* _index;
            
    //         _index = _inv_lists[i].index(value);
    
    //         vector<int> index;
    //         index.clear();
    //         if(_index != NULL)
    //             index = *_index;
    //         if(_inv_lists.size() <= 1)//used int subsequence search
    //             shift_bits_subsequence = _inv_lists[i]._shift_bits_subsequence();

    //         if (_ck.size() <= (unsigned int) key)
    //         {
    //             last = _ck.size();
    //             _ck.resize(key + 1);
    //             _inv_index.resize(key + 1);
    //             for (; last < _ck.size(); ++last)
    //             {
    //                 _ck[last] = _inv.size();
    //                 _inv_index[last] = _inv_pos.size();
    //             }
    //         }
    //         for (unsigned int j = 0; j < index.size(); ++j)
    //         {
    //             if (j % max_length == 0)
    //             {
    //                 _inv_pos.push_back(_inv.size());
    //             }
    //             _inv.push_back(index[j]);
    //             _ck[key] = _inv.size();
    //         }

    //     }

    // }
    // _inv_index.push_back(_inv_pos.size());
    // _inv_pos.push_back(_inv.size());

    // _build_status = builded;
    //     u64 table_end = getTime();
    // cout<<"build table time = "<<getInterval(table_start, table_end)<<"ms."<<endl;
    //     //Logger::log(Logger::DEBUG, "inv_index size %d:", _inv_index.size());
    // //Logger::log(Logger::DEBUG, "inv_pos size %d:", _inv_pos.size());
    // //Logger::log(Logger::DEBUG, "inv size %d:", _inv.size());
    //     Logger::log(Logger::INFO, "inv_index size %d:", _inv_index.size());
    // Logger::log(Logger::INFO, "inv_pos size %d:", _inv_pos.size());
    // Logger::log(Logger::INFO, "inv size %d:", _inv.size());

}

