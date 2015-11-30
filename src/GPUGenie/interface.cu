/*
 * interface.cpp
 *
 *  Created on: Jul 8, 2015
 *      Author: luanwenhao
 */

#include "interface.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <ctime>
#include <map>
#include <vector>
#include <algorithm>
#include <thrust/system_error.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace GPUGenie;
using namespace std;

namespace GPUGenie
{
	void
	load_table(inv_table& table, std::vector<std::vector<int> >& data_points, int max_length)
	{
		  inv_list list;
		  u32 i,j;
#ifdef GPUGENIE_DEBUG
		  printf("Data row size: %d. Data Row Number: %d.\n", data_points[0].size(), data_points.size());
		  u64 starttime = getTime();
#endif
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
#ifdef GPUGENIE_DEBUG
		  u64 endtime = getTime();
		  double timeInterval = getInterval(starttime, endtime);
		  printf("Before finishing loading. i_size():%d, m_size():%d.\n", table.i_size(), table.m_size());
		  cout<<">>>>[time profiling]: loading index takes "<<timeInterval<<" ms<<<<"<<endl;
#endif
	}
    
    void load_table(inv_table& table, int *data, unsigned int item_num, unsigned int *index, unsigned int row_num, int max_length)
    {
          inv_list list;
          u32 i,j;
#ifdef GPUGENIE_DEBUG
          printf("Data row size: %d. Data Row Number: %d. \n", index[1], row_num);
          u64 starttime = getTime();
#endif
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

#ifdef GPUGENIE_DEBUG
          u64 endtime = getTime();
          double timeInterval = getInterval(starttime, endtime);
          printf("Before finishing loading. i_size() : %d, m_size() : %d.\n", table.i_size(), table.m_size());
          cout<<">>>>[time profiling]: loading index takes "<<timeInterval<<" ms<<<<"<<endl;
#endif
    }


	void
	load_query(inv_table& table,
				std::vector<query>& queries,
				GPUGenie_Config& config)
	{
#ifdef GPUGENIE_DEBUG
		printf("Table dim: %d.\n", table.m_size());
		u64 starttime = getTime();
#endif
		u32 i,j;
		int value;
		int radius = config.query_radius;
		std::vector<std::vector<int> >& query_points = *config.query_points;
		for(i = 0; i < query_points.size(); ++i)
		{
			query q(table, i);

			for(j = 0; j < query_points[i].size() && j < config.dim; ++j){

				//for debug
				if(j>=12){
				value = query_points[i][j];
				if(value < 0)
				{
					continue;
				}
				q.attr(j,
					   value - radius < 0 ? 0 : value - radius,
					   value + radius,
					   GPUGENIE_DEFAULT_WEIGHT);
				}//end for debug
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
#ifdef GPUGENIE_DEBUG
		u64 endtime = getTime();
		double timeInterval = getInterval(starttime,endtime);
		printf("%d queries are created!\n", queries.size());
		cout<<">>>>[time profiling]: loading query takes "<<timeInterval<<" ms<<<<"<<endl;
#endif
	}
	void
	load_query_bijectMap(inv_table& table,
				std::vector<query>& queries,
				GPUGenie_Config& config)
	{
#ifdef GPUGENIE_DEBUG
		u64 starttime = getTime();
		printf("Table dim: %d.\n", table.m_size());
#endif
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
#ifdef GPUGENIE_DEBUG
		u64 endtime = getTime();
		double timeInterval = getInterval(starttime, endtime);
		printf("%d queries are created!\n", queries.size());
		cout<<">>>>[time profiling]: loading query takes "<<timeInterval<<" ms (for one dim multi-values)<<<<"<<endl;
#endif
	}

	void
	load_table_bijectMap(inv_table& table, std::vector<std::vector<int> >& data_points, int max_length)
	{

#ifdef GPUGENIE_DEBUG
	  u64 starttime = getTime();

#endif
	  inv_list list;
	  list.invert_bijectMap(data_points);
	  table.append(list);
	  table.build(max_length);
#ifdef GPUGENIE_DEBUG
	  u64 endtime = getTime();
	  double timeInterval = getInterval(starttime,endtime);
	  printf("Before finishing loading. i_size():%d, m_size():%d.\n", table.i_size(), table.m_size());
	  cout<<">>>>[time profiling]: loading index takes "<<timeInterval<<" ms (for one dim multi-values)<<<<"<<endl;
#endif
	}

    void
    load_table_bijectMap(inv_table& table, int *data, unsigned int item_num, unsigned int *index, 
                        unsigned int row_num, int max_length)
    {
#ifdef GPUGENIE_DEBUG
        u64 starttime = getTime();
#endif
        inv_list list;
        //list.invert_bijectMap(data_points);
        cout<<"s1"<<endl;
        list.invert_bijectMap(data, item_num, index, row_num);
        cout<<"s2"<<endl;
        table.append(list);
        cout<<"s3"<<endl;
        table.build(max_length);
#ifdef GPUGENIE_DEBUG
        u64 endtime = getTime();
        double timeInterval = getInterval(starttime, endtime);
        printf("Before finishing loading. i_size():%d, m_size():%d.\n", table.i_size(), table.m_size());
        cout<<">>>>[time profiling]: loading index takes "<<timeInterval<<" ms (for one dim multi-values)<<<<"<<endl;
#endif
    }

}
void GPUGenie::knn_search(std::vector<std::vector<int> >& data_points,
					  std::vector<std::vector<int> >& query_points,
					  std::vector<int>& result,
					  int num_of_topk)
{
	knn_search(data_points,
			   query_points,
			   result,
			   num_of_topk,
			   GPUGENIE_DEFAULT_RADIUS,
			   GPUGENIE_DEFAULT_THRESHOLD,
			   GPUGENIE_DEFAULT_HASHTABLE_SIZE,
			   GPUGENIE_DEFAULT_DEVICE);
}
void GPUGenie::knn_search(std::vector<std::vector<int> >& data_points,
					  std::vector<std::vector<int> >& query_points,
					  std::vector<int>& result,
					  int num_of_topk,
					  int radius,
					  int threshold,
					  float hashtable,
					  int device)
{
	GPUGenie_Config config;
	config.count_threshold = threshold;
	config.data_points = &data_points;
	config.hashtable_size = hashtable;
	config.num_of_topk = num_of_topk;
	config.query_points = &query_points;
	config.query_radius = radius;
	config.use_device = device;
	if(!query_points.empty())
		config.dim = query_points[0].size();

	knn_search(result, config);
}

void GPUGenie::knn_search_bijectMap(std::vector<int>& result, GPUGenie_Config& config)
{
	inv_table table;
	std::vector<query> queries;

#ifdef GPUGENIE_DEBUG
	printf("Building table...");
#endif
    if(config.item_num ==0){
	load_table_bijectMap(table, *(config.data_points), config.posting_list_max_length);
    }else if(config.data != NULL && config.index != NULL && config.item_num!=0 && config.row_num!=0){
        load_table_bijectMap(table, config.data, config.item_num, config.index, config.row_num, config.posting_list_max_length);
    }else{
        printf("no data input!\n");
        return;
    }
#ifdef GPUGENIE_DEBUG
	printf("Done!\n");
	printf("Loading queries...");
#endif

#ifdef GPUGENIE_DEBUG
	u64 starttime = getTime();
#endif

	load_query_bijectMap(table,queries,config);

#ifdef GPUGENIE_DEBUG
	printf("Done!\n");
#endif

	knn_search(table, queries, result, config);
#ifdef GPUGENIE_DEBUG
	u64 endtime = getTime();
	double elapsed = getInterval(starttime, endtime);
	 cout<<">>>>[time profiling]: knn_search totally takes "<<elapsed<<" ms (building query+match+selection)<<<<"<<endl;
#endif
}

void GPUGenie::knn_search(std::vector<int>& result, GPUGenie_Config& config)
{
	inv_table table;
	std::vector<query> queries;
	#ifdef GPUGENIE_DEBUG
	printf("Building table...");
#endif
    
    if(config.item_num == 0){
	    cout<<"build from data_points..."<<endl;
	    load_table(table, *(config.data_points), config.posting_list_max_length);
    }else if(config.data != NULL&&config.index!=NULL && config.item_num!=0&& config.row_num!=0){
	cout<<"build from data array..."<<endl;
        load_table(table, config.data, config.item_num, config.index, config.row_num, config.posting_list_max_length);
    }else{
        printf("no data input!\n");
        return;
    }

#ifdef GPUGENIE_DEBUG
	printf("Done!\n");
	printf("Loading queries...");
    u64 starttime = getTime();
#endif

	load_query(table,queries,config);

#ifdef GPUGENIE_DEBUG
	printf("Done!\n");
#endif

	knn_search(table, queries, result, config);


#ifdef GPUGENIE_DEBUG
	u64 endtime = getTime();
	double elapsed = getInterval(starttime, endtime);
	 cout<<">>>>[time profiling]: knn_search totally takes "<<elapsed<<" ms (building query+match+selection)<<<<"<<endl;
#endif
}


void GPUGenie::knn_search(inv_table& table,
					  std::vector<query>& queries,
					  std::vector<int>& h_topk,
					  GPUGenie_Config& config)
{
	int device_count, hashtable_size;
	cudaGetDeviceCount(&device_count);
	if(device_count == 0)
	{
		printf("[Info] NVIDIA CUDA-SUPPORTED GPU NOT FOUND!\nProgram aborted..\n");
		exit(2);
	} else if(device_count <= config.use_device)
	{
#ifdef GPUGENIE_DEBUG
		printf("[Info] Device %d not found!", config.use_device);
#endif
		config.use_device = GPUGENIE_DEFAULT_DEVICE;
	}
	cudaSetDevice(config.use_device);
#ifdef GPUGENIE_DEBUG
	printf("Using device %d...\n", config.use_device);
	printf("table.i_size():%d, config.hashtable_size:%f.\n", table.i_size(), config.hashtable_size);
#endif
	if(config.hashtable_size<=2){
		hashtable_size = table.i_size() * config.hashtable_size + 1;
	}else{
		hashtable_size =  config.hashtable_size;
	}
	thrust::device_vector<int> d_topk;

#ifdef GPUGENIE_DEBUG
	printf("Starting knn search...\n");
#endif
	int max_load = config.multiplier * config.posting_list_max_length + 1;
	printf("max_load is %d\n", max_load);
	GPUGenie::knn_bijectMap(table,//basic API, since encode dimension and value is also finally transformed as a bijection map
			   queries,
			   d_topk,
			   hashtable_size,
			   max_load,
			   config.count_threshold,
			   config.dim,
			   config.num_of_hot_dims,
			   config.hot_dim_threshold);

#ifdef GPUGENIE_DEBUG
	printf("knn search is done!\n");
	printf("Topk obtained: %d in total.\n", d_topk.size());
#endif

	h_topk.resize(d_topk.size());
	thrust::copy(d_topk.begin(), d_topk.end(), h_topk.begin());
}



