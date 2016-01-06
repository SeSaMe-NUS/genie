/*
 * interface.cpp
 *
 *  Created on: Jul 8, 2015
 *      Author: luanwenhao
 */

#include "interface.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <string>
#include <sys/time.h>
#include <ctime>
#include <map>
#include <vector>
#include <algorithm>

#include <thrust/system_error.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Logger.h"

using namespace GPUGenie;
using namespace std;

namespace GPUGenie
{

	void
	load_table(inv_table& table, std::vector<std::vector<int> >& data_points, int max_length, bool save_to_gpu)
	{
		  inv_list list;
		  u32 i,j;

		  Logger::log(Logger::DEBUG, "Data row size: %d. Data Row Number: %d.", data_points[0].size(), data_points.size());
		  u64 starttime = getTime();

		  for(i = 0; i < data_points[0].size(); ++i)
		  {
			  std::vector<int> col;
			  col.reserve(data_points.size());
			  for(j = 0; j < data_points.size(); ++j)
			  {
				  col.push_back(data_points[j][i]);
			  }
			  list.invert(col);
			  table.append(list);
		  }

		  table.build(max_length);
          if(save_to_gpu == true)
          {
             device_vector<int> d_ck(*table.ck());
             table.d_ck_p = raw_pointer_cast(d_ck.data());
             
             device_vector<int> d_inv(*table.inv());
             table.d_inv_p = raw_pointer_cast(d_inv.data());
             
             device_vector<int> d_inv_index(*table.inv_index());
             table.d_inv_index_p = raw_pointer_cast(d_inv_index.data());
             
             device_vector<int> d_inv_pos(*table.inv_pos());
             table.d_inv_pos_p = raw_pointer_cast(d_inv_pos.data());

          }
          table.is_stored_in_gpu = save_to_gpu;

		  u64 endtime = getTime();
		  double timeInterval = getInterval(starttime, endtime);
		  Logger::log(Logger::DEBUG, "Before finishing loading. i_size():%d, m_size():%d.", table.i_size(), table.m_size());
		  Logger::log(Logger::VERBOSE, ">>>>[time profiling]: loading index takes %f ms<<<<",timeInterval);

	}
    
    void load_table(inv_table& table, int *data, unsigned int item_num, unsigned int *index, unsigned int row_num, int max_length, bool save_to_gpu)
    {
          inv_list list;
          u32 i,j;

          Logger::log(Logger::DEBUG,"Data row size: %d. Data Row Number: %d.", index[1], row_num);
          u64 starttime = getTime();

          for(i = 0; i<index[1]; ++i){
              std::vector<int> col;
	      col.reserve(row_num);
              for(j = 0; j<row_num; ++j){
                col.push_back(data[index[j]+i]);
             }
	      list.invert(col);
              table.append(list);
          }
          table.build(max_length);

          if(save_to_gpu == true)
          {
             device_vector<int> d_ck(*table.ck());
             table.d_ck_p = raw_pointer_cast(d_ck.data());
             
             device_vector<int> d_inv(*table.inv());
             table.d_inv_p = raw_pointer_cast(d_inv.data());
             
             device_vector<int> d_inv_index(*table.inv_index());
             table.d_inv_index_p = raw_pointer_cast(d_inv_index.data());
             
             device_vector<int> d_inv_pos(*table.inv_pos());
             table.d_inv_pos_p = raw_pointer_cast(d_inv_pos.data());

          }
          table.is_stored_in_gpu = save_to_gpu;

          u64 endtime = getTime();
          double timeInterval = getInterval(starttime, endtime);
          Logger::log(Logger::DEBUG,"Before finishing loading. i_size() : %d, m_size() : %d.", table.i_size(), table.m_size());
          Logger::log(Logger::VERBOSE,">>>>[time profiling]: loading index takes %f ms<<<<",timeInterval);

    }

    //Read new format query data
    //Sample data format
    //qid dim value selectivity weight
    // 0   0   15     0.04        1
    // 0   1   6      0.04        1
    // ....
	void load_query_multirange(inv_table& table,
							   std::vector<query>& queries,
							   GPUGenie_Config& config)
    {
		queries.clear();
		map<int, query> query_map;
		int qid,dim,val;
		float sel, weight;
		for (int iq = 0; iq < config.multirange_query_points->size(); ++iq) {
			attr_t& attr = (*config.multirange_query_points)[iq];

			qid = attr.qid;
			dim = attr.dim;
			val = attr.value;
			weight = attr.weight;
			sel = attr.sel;
			if(query_map.find(qid) == query_map.end()){
				query q(table, qid);
				q.topk(config.num_of_topk);
				if(config.selectivity > 0.0f)
				{
					q.selectivity(config.selectivity);
				}
				if(config.use_load_balance)
				{
					q.use_load_balance = true;
				}
				query_map[qid]= q;

			}
			query_map[qid].attr(dim,val, weight, sel);
		}
		for(std::map<int, query>::iterator it = query_map.begin();
			it != query_map.end() && queries.size() < config.num_of_queries;
			++it)
		{
			query& q = it->second;
			q.apply_adaptive_query_range();
			queries.push_back(q);
		}

		Logger::log(Logger::INFO,"Finish loading queries! %d queries are loaded.", queries.size());

    }
	void
	load_query(inv_table& table,
				std::vector<query>& queries,
				GPUGenie_Config& config)
	{

		Logger::log(Logger::DEBUG,"Table dim: %d.", table.m_size());
		u64 starttime = getTime();

		u32 i,j;
		int value;
		int radius = config.query_radius;
		std::vector<std::vector<int> >& query_points = *config.query_points;
		for(i = 0; i < query_points.size(); ++i)
		{
			query q(table, i);

			for(j = 0; j < query_points[i].size() && j < config.dim; ++j){
				value = query_points[i][j];
				if(value < 0)
				{
					continue;
				}
				q.attr(j,
					   value - radius < 0 ? 0 : value - radius,
					   value + radius,
					   GPUGENIE_DEFAULT_WEIGHT);
			}

			q.topk(config.num_of_topk);
			q.selectivity(config.selectivity);
			if(config.use_adaptive_range)
			{
				q.apply_adaptive_query_range();
			}
			if(config.use_load_balance)
			{
				q.use_load_balance = true;
			}

			queries.push_back(q);
		}

		u64 endtime = getTime();
		double timeInterval = getInterval(starttime,endtime);
		Logger::log(Logger::INFO,"%d queries are created!", queries.size());
		Logger::log(Logger::VERBOSE,">>>>[time profiling]: loading query takes %f ms<<<<",timeInterval);

	}
	void
	load_query_bijectMap(inv_table& table,
				std::vector<query>& queries,
				GPUGenie_Config& config)
	{

		u64 starttime = getTime();
		Logger::log(Logger::DEBUG,"Table dim: %d.", table.m_size());

		u32 i,j;
		int value;
		int radius = config.query_radius;
		std::vector<std::vector<int> >& query_points = *config.query_points;
		for(i = 0; i < query_points.size(); ++i)
		{
			query q(table, i);

			for(j = 0; j < query_points[i].size(); ++j){

				value = query_points[i][j];
				if(value < 0)
				{
					continue;
				}
				q.attr(0,
					   value - radius < 0 ? 0 : value - radius,
					   value + radius,
					   GPUGENIE_DEFAULT_WEIGHT);
			}

			q.topk(config.num_of_topk);
			q.selectivity(config.selectivity);
			if(config.use_adaptive_range)
			{
				q.apply_adaptive_query_range();
			}
			if(config.use_load_balance)
			{
				q.use_load_balance = true;
			}

			queries.push_back(q);
		}

		u64 endtime = getTime();
		double timeInterval = getInterval(starttime, endtime);
		Logger::log(Logger::INFO,"%d queries are created!", queries.size());
		Logger::log(Logger::VERBOSE,">>>>[time profiling]: loading query takes %f ms (for one dim multi-values)<<<<",timeInterval);

	}

	void
	load_table_bijectMap(inv_table& table, std::vector<std::vector<int> >& data_points, int max_length, bool save_to_gpu)
	{


	  u64 starttime = getTime();

	  inv_list list;
	  list.invert_bijectMap(data_points);
	  table.append(list);
	  table.build(max_length);


      if(save_to_gpu == true)
      {
          device_vector<int> d_ck(*table.ck());
          table.d_ck_p = raw_pointer_cast(d_ck.data());
             
          device_vector<int> d_inv(*table.inv());
          table.d_inv_p = raw_pointer_cast(d_inv.data());
             
          device_vector<int> d_inv_index(*table.inv_index());
          table.d_inv_index_p = raw_pointer_cast(d_inv_index.data());
             
          device_vector<int> d_inv_pos(*table.inv_pos());
          table.d_inv_pos_p = raw_pointer_cast(d_inv_pos.data());
      }
      table.is_stored_in_gpu = save_to_gpu;

	  u64 endtime = getTime();
	  double timeInterval = getInterval(starttime,endtime);
	  Logger::log(Logger::DEBUG,"Before finishing loading. i_size():%d, m_size():%d.", table.i_size(), table.m_size());
	  Logger::log(Logger::VERBOSE,">>>>[time profiling]: loading index takes %f ms (for one dim multi-values)<<<<",timeInterval);

	}

    void
    load_table_bijectMap(inv_table& table, int *data, unsigned int item_num, unsigned int *index, 
                        unsigned int row_num, int max_length, bool save_to_gpu)
    {

        u64 starttime = getTime();

        inv_list list;
        list.invert_bijectMap(data, item_num, index, row_num);

        table.append(list);
        table.build(max_length);


        if(save_to_gpu == true)
        {
             device_vector<int> d_ck(*table.ck());
             table.d_ck_p = raw_pointer_cast(d_ck.data());
             
             device_vector<int> d_inv(*table.inv());
             table.d_inv_p = raw_pointer_cast(d_inv.data());
             
             device_vector<int> d_inv_index(*table.inv_index());
             table.d_inv_index_p = raw_pointer_cast(d_inv_index.data());
             
             device_vector<int> d_inv_pos(*table.inv_pos());
             table.d_inv_pos_p = raw_pointer_cast(d_inv_pos.data());
        }
        table.is_stored_in_gpu = save_to_gpu;

        u64 endtime = getTime();
        double timeInterval = getInterval(starttime, endtime);
        Logger::log(Logger::DEBUG,"Before finishing loading. i_size():%d, m_size():%d.", table.i_size(), table.m_size());
        Logger::log(Logger::VERBOSE,">>>>[time profiling]: loading index takes %f ms (for one dim multi-values)<<<<",timeInterval);

    }

}
void GPUGenie::knn_search_bijectMap(std::vector<int>& result, GPUGenie_Config& config)
{
	std::vector<int> result_count;
	knn_search_bijectMap(result, result_count, config);
}

void GPUGenie::knn_search_bijectMap(std::vector<int>& result, std::vector<int>& result_count, GPUGenie_Config& config)
{
	inv_table table;
	std::vector<query> queries;

	Logger::log(Logger::INFO,"Building table...");

    if(config.item_num ==0){
	load_table_bijectMap(table, *(config.data_points), config.posting_list_max_length);
    }else if(config.data != NULL && config.index != NULL && config.item_num!=0 && config.row_num!=0){
        load_table_bijectMap(table, config.data, config.item_num, config.index, config.row_num, config.posting_list_max_length);
    }else{
    	Logger::log(Logger::ALERT,"no data input!");
        return;
    }

    Logger::log(Logger::INFO,"Loading queries...");

	u64 starttime = getTime();

	load_query_bijectMap(table,queries,config);

	knn_search(table, queries, result, config);

	u64 endtime = getTime();
	double elapsed = getInterval(starttime, endtime);
	Logger::log(Logger::VERBOSE, ">>>>[time profiling]: knn_search totally takes %f ms (building query+match+selection)<<<<",elapsed);
}

void GPUGenie::knn_search(std::vector<int>& result,
						  GPUGenie_Config& config)
{
	std::vector<int> result_count;
	knn_search(result, result_count, config);
}

void GPUGenie::knn_search(std::vector<int>& result,
						  std::vector<int>& result_count,
						  GPUGenie_Config& config)
{
	inv_table table;
	std::vector<query> queries;

	Logger::log(Logger::INFO,"Building table...");
    
    if(config.item_num == 0){
    	Logger::log(Logger::INFO,"build from data_points...");
	    load_table(table, *(config.data_points), config.posting_list_max_length);
    }else if(config.data != NULL&&config.index!=NULL && config.item_num!=0&& config.row_num!=0){
    	Logger::log(Logger::INFO,"build from data array...");
        load_table(table, config.data, config.item_num, config.index, config.row_num, config.posting_list_max_length);
    }else{
    	Logger::log(Logger::ALERT,"no data input!");
        return;
    }

    Logger::log(Logger::INFO,"Loading queries...");
    u64 starttime = getTime();

    if(config.use_multirange){
    	load_query_multirange(table,queries,config);
    } else {
    	load_query(table,queries,config);
    }

    Logger::log(Logger::INFO,"Starting knn_search ...");

	knn_search(table, queries, result, result_count, config);



	u64 endtime = getTime();
	double elapsed = getInterval(starttime, endtime);

	Logger::log(Logger::VERBOSE,">>>>[time profiling]: knn_search totally takes %f ms (building query+match+selection)<<<<",elapsed);
}

void GPUGenie::knn_search(inv_table& table,
					  std::vector<query>& queries,
					  std::vector<int>& h_topk,
					  GPUGenie_Config& config)
{
	std::vector<int> h_topk_count;
	knn_search(table, queries, h_topk, h_topk_count, config);
}
void GPUGenie::knn_search(inv_table& table,
					  std::vector<query>& queries,
					  std::vector<int>& h_topk,
					  std::vector<int>& h_topk_count,
					  GPUGenie_Config& config)
{
	int device_count, hashtable_size;
	cudaGetDeviceCount(&device_count);
	if(device_count == 0)
	{
		Logger::log(Logger::ALERT,"[ALERT] NVIDIA CUDA-SUPPORTED GPU NOT FOUND! Program aborted..");
		exit(2);
	} else if(device_count <= config.use_device)
	{
		Logger::log(Logger::INFO,"[Info] Device %d not found! Changing to %d...", config.use_device, GPUGENIE_DEFAULT_DEVICE);
		config.use_device = GPUGENIE_DEFAULT_DEVICE;
	}
	cudaSetDevice(config.use_device);

	Logger::log(Logger::INFO,"Using device %d...", config.use_device);
	Logger::log(Logger::DEBUG,"table.i_size():%d, config.hashtable_size:%f.", table.i_size(), config.hashtable_size);

	if(config.hashtable_size<=2){
		hashtable_size = table.i_size() * config.hashtable_size + 1;
	}else{
		hashtable_size =  config.hashtable_size;
	}
	thrust::device_vector<int> d_topk, d_topk_count;

	int max_load = config.multiplier * config.posting_list_max_length + 1;

	Logger::log(Logger::DEBUG,"max_load is %d", max_load);

	GPUGenie::knn_bijectMap(table,//basic API, since encode dimension and value is also finally transformed as a bijection map
			   queries,
			   d_topk,
			   d_topk_count,
			   hashtable_size,
			   max_load,
			   config.count_threshold,
			   config.dim);


	Logger::log(Logger::INFO,"knn search is done!");
	Logger::log(Logger::DEBUG,"Topk obtained: %d in total.", d_topk.size());


	h_topk.resize(d_topk.size());
	h_topk_count.resize(d_topk_count.size());
	thrust::copy(d_topk.begin(), d_topk.end(), h_topk.begin());
	thrust::copy(d_topk_count.begin(), d_topk_count.end(), h_topk_count.begin());
}



