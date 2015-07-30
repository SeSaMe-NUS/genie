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

using namespace GaLG;
using namespace std;

namespace GaLG
{
	void
	load_table(inv_table& table, std::vector<std::vector<int> >& data_points)
	{
		  inv_list list;
		  u32 i,j;
		  std::vector<int> col;

		  for(i = 0; i < data_points[0].size(); ++i)
		  {
			  col.resize(data_points.size());
			  for(j = 0; j < data_points.size(); ++j)
			  {
				  col.push_back(data_points[j][i]);
			  }
			  list.invert(col);
			  table.append(list);
			  col.clear();
		  }

		  table.build();
	}

	void
	load_query(inv_table& table,
				std::vector<query> queries,
				GaLG_Config& config)
	{
		u32 i,j;
		int value;
		int radius = config.query_radius;
		std::vector<std::vector<int> >& query_points = *config.query_points;
		for(i = 0; i < query_points.size(); ++i)
		{
			query q(table);
			for(j = 0; j < query_points[i].size(); ++i){
				value = query_points[i][j];
				q.attr(j,
					   value - radius < 0 ? 0 : value - radius,
					   value + radius,
					   GALG_DEFAULT_WEIGHT);
			}
			q.topk(config.num_of_topk);
			q.selectivity(config.selectivity);
			if(config.use_adaptive_range)
			{
				q.apply_adaptive_query_range();
			}
			queries.push_back(q);
		}
	}
}
void GaLG::knn_search(std::vector<std::vector<int> >& data_points,
					  std::vector<std::vector<int> >& query_points,
					  std::vector<int>& result,
					  int num_of_topk)
{
	knn_search(data_points,
			   query_points,
			   result,
			   num_of_topk,
			   GALG_DEFAULT_RADIUS,
			   GALG_DEFAULT_THRESHOLD,
			   GALG_DEFAULT_HASHTABLE_SIZE,
			   GALG_DEFAULT_DEVICE);
}
void GaLG::knn_search(std::vector<std::vector<int> >& data_points,
					  std::vector<std::vector<int> >& query_points,
					  std::vector<int>& result,
					  int num_of_topk,
					  int radius,
					  int threshold,
					  float hashtable,
					  int device)
{
	GaLG_Config config;
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

void GaLG::knn_search(std::vector<int>& result, GaLG_Config& config)
{
	inv_table table;
	std::vector<query> queries;
	int hashtable_size = config.hashtable_size * table.i_size() + 1;

	printf("Building table...");
	load_table(table, *(config.data_points));
	printf("Done!\n");

	printf("Loading queries...");
	load_query(table,queries,config);
	printf("Done!\n");

	knn_search(table, queries, result, config);
}


void GaLG::knn_search(inv_table& table,
					  std::vector<query>& queries,
					  std::vector<int>& h_topk,
					  GaLG_Config& config)
{
	int device_count, hashtable_size;
	cudaGetDeviceCount(&device_count);
	if(device_count == 0)
	{
		printf("[Info] NVIDIA CUDA-SUPPORTED GPU NOT FOUND!\nProgram aborted..\n");
		return;
	} else if(device_count <= config.use_device)
	{
		printf("[Info] Device %d not found!", config.use_device);
		config.use_device = GALG_DEFAULT_DEVICE;
	}
	cudaSetDevice(config.use_device);
	cudaDeviceReset();
	cudaDeviceSynchronize();
	printf("Using device %d...\n", config.use_device);

	hashtable_size = table.i_size() * config.hashtable_size + 1;
	thrust::device_vector<int> d_topk;

	printf("Starting knn search...\n");
	GaLG::topk(table,
			   queries,
			   d_topk,
			   hashtable_size,
			   config.count_threshold,
			   config.dim,
			   config.num_of_hot_dims,
			   config.hot_dim_threshold);
	printf("knn search is done!\n");

	printf("Topk obtained: %d in total.\n", d_topk.size());
	h_topk.resize(d_topk.size());

	thrust::copy(d_topk.begin(), d_topk.end(), h_topk.begin());
}



