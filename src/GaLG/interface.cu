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

using namespace GaLG;
using namespace std;


void
load_table(inv_table& table, std::vector<std::vector<int> >& data_points)
{
	  inv_list list;
	  u32 i,j;
	  std::vector<int> col;

	  for(i = 0; i < num_of_dims; ++i)
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
			std::vector<std::vector<int> > query_points,
		    std::vector<query> queries,
		    int num_of_topk,
		    int radius)
{
	u32 i,j;
	int value;
	queries.clear();
	queries.resize(query_points.size());
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
		q.topk(num_of_topk);
		queries.push_back(q);
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

void GaLG::knn_search(std::vector<int>& result, GaLG_Config& config)
{
	knn_search(config.data_points,
			   config.query_points,
			   result,
			   config.num_of_topk,
			   config.query_radius,
			   config.count_threshold,
			   config.hashtable_size,
			   config.use_device);
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
	inv_table table;
	std::vector<query> queries;
	int hashtable_size = hashtable * table.i_size() + 1;
	int device_count;

	printf("Building table...");
	load_table(table, data_points);
	printf("Done!\n");

	printf("Loading queries...");
	load_query(table, query_points, queries, num_of_topk, radius);
	printf("Done!\n");

	cudaGetDeviceCount(&device_count);
	if(device_count == 0)
	{
		printf("[Info] NVIDIA CUDA-SUPPORTED GPU NOT FOUND!\nProgram aborted..\n");
		return;
	} else if(device_count <= device)
	{
		printf("[Info] Device %d not found!", device);
		device = GALG_DEFAULT_DEVICE;
		printf("Using device %d...\n", device);
	}
	cudaSetDevice(device);
	cudaDeviceReset();
	cudaDeviceSynchronize();

	device_vector<int> d_topk;






}
void GaLG::knn_search(inv_table& table,
				std::vector<query>& queries,
				int radius,
				int bitmap_bits,
				float hash_table_size_)
{
	 cudaDeviceReset();
	 int device_count;
	 cudaGetDeviceCount(&device_count);
	 cudaSetDevice(device_count - 1);

	  u64 timestart, timestop, totalstart;

	  totalstart = timestart = getTime();

	  device_vector<int> d_topk;
	  int hash_table_size = hash_table_size_ * table.i_size() + 1;
	  printf("hash table size: %d\n", hash_table_size);

	  timestart = getTime();
	  GaLG::topk(table, queries, d_topk, hash_table_size, bitmap_bits, num_of_query_dims);
	  if(GALG_ERROR){
		  cudaDeviceReset();
		  return;
	  }
	  timestop = getTime();
	  GALG_TIME += (timestop - timestart);
	  printf("Topk takes %f ms.\n", getInterval(timestart, timestop));

	  printf("Starting copying device result to host...\n");
	  timestart = getTime();
	  host_vector<int> h_topk(d_topk);
	  cudaDeviceSynchronize();
	  timestop = getTime();
	  printf("Finish copying result. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  for(int i = 0; i < queries.size() && i < num_of_query_print; ++i)
	  {
		  printf("The top %d of query %d is \n", queries[i].topk(),i);
		  if(queries[i].topk() > 0) printf("%d", h_topk[0 + i * queries[i].topk()]);
		  for(int j = 1; j < queries[i].topk(); ++j)
		  {
			  printf(", %d", h_topk[i * queries[i].topk() + j]);
		  }
		  printf("\n");
	  }

	  printf(">>>>>>>>>>>>>Successful topk searching.\n");
	  timestop = getTime();
	  printf("Finish testing. Time elapsed: %f ms. \n", getInterval(totalstart, timestop));
}



std::string get_cmd_option(std::vector<std::string>::iterator& begin, std::vector<std::string>::iterator& end, const std::string & option)
{
	std::vector<std::string>::iterator itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmd_option_exists(std::vector<std::string>::iterator& begin, std::vector<std::string>::iterator& end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int stoi(std::string str)
{
	int result = atoi(str.c_str());
	if(str.empty() ||(eraseSpace(str) != std::string("0") && result == 0)){
		throw 0;
	}
	return result;
}

float stof(std::string str)
{
	float result = atof(str.c_str());
	if(str.empty() ||  ( eraseSpace(str) != std::string("0") && result == 0.0f)){
		throw 0;
	}
	return result;
}


