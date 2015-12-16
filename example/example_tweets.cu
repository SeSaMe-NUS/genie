/*
 * example_tweets.cu
 *
 *  Created on: Oct 16, 2015
 *      Author: zhoujingbo
 *
 * description: This program is to demonstrate the search on string-like data by the GPU. More description of the parameter configuration please refer to example.cu file
 */

#include "GPUGenie.h" //for ide: change from "GPUGenie.h" to "../src/GPUGenie.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>

using namespace GPUGenie;
using namespace std;


int main(int argc, char * argv[])//for ide: from main to main4
{
	std::vector<std::vector<int> > queries;
	std::vector<std::vector<int> > data;
	inv_table table;
	
	read_file(data, "tweets_4k.csv", -1);//for debug
	read_file(queries, "tweets_4k.csv", 100);//for debug
	GPUGenie::GPUGenie_Config config;

	//Data dimension
	//For string search, we use one-dimension-mulitple-values method,
	// i.e. there is only one dimension, all words are considered as the values. 
	//given a query, there can be multiple values for this dimension. 
	//This is like a bag-of-word model for string search
	config.dim = 14;

	//Points with dim counts lower than threshold will be discarded and not shown in topk.
	//It is implemented as a bitmap filter.
	//Set to 0 to disable the feature.
	//set to <0, to use adaptiveThreshold, the absolute value of count_threshold is the maximum possible count sotred in the bitmap
	config.count_threshold = -14;

	//Number of topk items desired for each query.
	config.num_of_topk = 100;

	//if config.hashtable_size<=2, the hashtable_size means ratio against data size
			//Hash Table size is set as: config.hashtable_size (i.e.) ratio X data size.
			//Topk items will be generated from the hash table so it must be sufficiently large.
			//If set too small, the program will attempt to increase the size by 0.1f as many times
			//as possible. So to reduce the attempt time waste, please set to 1.0f if memory allows.
	//if config.hashtable_size>2, the hashtable_size means the size of the hashtable,
			//this is useful when using adaptiveThreshold (i.e. config.count_threshold <0), where the
			//hash_table size is usually set as: maximum_countXconfig.num_of_topkx1.5 (where 1.5 is load factor for hashtable).
	config.hashtable_size = 14*config.num_of_topk*1.5;//960

	//Query radius from the data point bucket expanding to upward and downward.
		//For tweets data, set it as 0, which means exact match
	//Will be overwritten by selectivity if use_adaptive_range is set.
	config.query_radius = 0;

	//Index of the GPU device to be used. If you only have one card, then set to 0.
	config.use_device = 0;

	//use_adaptive_range is not suitable for string search
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;

	config.data_points = &data;
	config.query_points = &queries;

	//if use_load_balance=false, config.multiplier and config.posting_list_max_length are not useful
	config.use_load_balance = true;
	//maximum number per posting list, if a keyword has a long posting list, we break it into sublists, and this parameter defines the maximum length of sub-list
	config.posting_list_max_length = 64000;
	config.multiplier = 1.5f;//config.multiplier*config.posting_list_max_length is  maximum number of elements processed by one block

	std::vector<int> result;
	printf("Launching knn functions...\n");
	u64 start = getTime();

	/**
	* @brief Search on the inverted index and save the result in result
	* bijectMap means building each ordered pair/keyword is also transformed by a bijection map. (Different from the default method, where the
	* keyword is a combination of dimension and value
	* Previous name: knn_search_tweets()
	*
	*/
	GPUGenie::knn_search_bijectMap(result, config);
	u64 end = getTime();
	double elapsed = getInterval(start, end);
	printf(">>>>>>>[time profiling]: Total Time Elapsed: %fms. <<<<<<<\n", elapsed);

	for(int i = 0; i < 10 && i < queries.size(); ++i)
	{
		printf("Query %d result is: \n\t", i);
		for (int j = 0; j < 10; ++j)
		{
			printf("%d, ", result[i * config.num_of_topk + j]);
		}
		printf("\n");
	}
}


