/**
*example clean and format by jingbo
description: create a running example fo the library. 
2015.09.10
*/

#include "GPUGenie.h" //for ide: change from "GPUGenie.h" to "../src/GPUGenie.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>

using namespace GPUGenie;
using namespace std;


int main(int argc, char * argv[])
{

	Logger::set_level(Logger::DEBUG);

	std::vector<std::vector<int> > queries;
	std::vector<attr_t> multirange_queries;
	std::vector<std::vector<int> > data;
	inv_table table;

	//Reading file from the disk. Alternatively, one can simply use vectors generated from other functions
	//Example vectors:
	//Properties: 10 points, 5 dimensions, value range 0-255, -1 for excluded dimensions.
	//|id|dim0|dim1|dim2|dim3|dim4|
	//|0	|2	 |255 |16  |0   |-1  |
	//|1	|10	 |-1  |52  |62  |0   |
	//|...  |... |... |... |... |... |
	//|9	|0   |50  |253 |1   |164 |

	string  dataFile = "sift_1k.csv";//for ide: from "sift_1k.csv" to "example/sift_1k.csv"
	string queryFile = "sift_1k_query.csv";
	
	/*** Configuration of KNN Search ***/
	GPUGenie::GPUGenie_Config config;

	config.num_of_queries = 10;

	//Data dimension
	config.dim = 128;

	//Points with dim counts lower than threshold will be discarded and not shown in topk.
	//It is implemented as a bitmap filter.
	//Set to 0 to disable the feature.
	//set to <0, to use adaptiveThreshold, the absolute value of count_threshold is the maximum possible count sotred in the bitmap
	config.count_threshold = -128;

	//Number of topk items desired for each query.
	//Some queries may result in fewer than desired topk items.
	config.num_of_topk = 10;

	//if config.hashtable_size<=2, the hashtable_size means ratio against data size
		//Hash Table size is set as: config.hashtable_size (i.e.) ratio X data size.
		//Topk items will be generated from the hash table so it must be sufficiently large.
		//If set too small, the program will attempt to increase the size by 0.1f as many times
		//as possible. So to reduce the attempt time waste, please set to 1.0f if memory allows.
	//if config.hashtable_size>2, the hashtable_size means the size of the hashtable,
		//this is useful when using adaptiveThreshold (i.e. config.count_threshold <0), where the
		//hash_table size is usually set as: maximum_countXconfig.num_of_topkx1.5 (where 1.5 is load factor for hashtable).
	config.hashtable_size = 128*config.num_of_topk*1.5;//960




	//Query radius from the data point bucket expanding to upward and downward.
	//Will be overwritten by selectivity if use_adaptive_range is set.
	config.query_radius = 1;

	//Index of the GPU device to be used. If you only have one card, then set to 0.
	config.use_device = 1;


	//Set if adaptive range of query is used.
	//Once set with a valid selectivity, the query will be re-scanned to
	//guarantee at least (selectivity * data size) of the data points will be matched
	//for each dimension.
	config.use_adaptive_range = true;

	//The selectivity to be used. Range 0.0f (no other bucket to be matched) to 1.0f (match all buckets).

	config.selectivity = 0.2f;


	//The pointer to the vector containing the data.
	config.data_points = &data;

	//The pointer to the vector containing the queries.


	config.multiplier = 1.5f;
	config.posting_list_max_length = 6400;
	config.use_load_balance = true;

	config.use_multirange = true;

	read_file(data, dataFile.c_str(), -1);//for AT: for adaptiveThreshold
	if(config.use_multirange)
	{
		read_query(multirange_queries, queryFile.c_str(), -1);
		config.multirange_query_points = &multirange_queries;
	} else {
		read_file(queries, queryFile.c_str(), config.num_of_queries);
		config.query_points = &queries;
	}


	/*** End of Configuration ***/

	/*** NOTE TO DEVELOPERS ***/
	/*
	 The library is still under development. Therefore there might be crashes if
	 the parameters are not set optimally.

	 Optimal settings may help you launch ten times more queries than naive setting,
	 but reducing hash table size may result in unexpected crashes or insufficient
	 topk results. (How can you expect 100 topk results from a hash table of size 50
	 or a threshold of 100 suppose you only have 128 dimensions?) So please be careful
	 if you decide to use non-default settings.

	 The basic & must-have settings are:
	 1. dim
	 2. num_of_topk
	 3. data_points
	 4. query_points
	 And leave the rest to default.

	 Recommended settings:
	 If you want to increase the query selectivity (default is one bucket itself, i.e. radius 0,
	 which is absolutely not enough), you can use either way as follows:
	 A. set radius to a larger value, e.g. 1, 2, 3 and so on. It will expand your search
	 	 upwards and downward by the amount of buckets you set. However, it does not guarantee
	 	 the selectivity since data distribution may not even.
	 B. set the selectivity and turn on use_adaptive_range, e.g. selectivity = 0.01 and
	 	 use_adaptive_range = true. It can guarantee that on each dimension the query will match
	 	 for at least (selectivity * data size) number of data points.
	 Note that approach B will overwrite approach A if both are set.

	 Advanced settings:
	 For num_of_hot_dims and hot_dim_threshold, please contact the author for details.

	 Author: Luan Wenhao
	 Contact: lwhluan@gmail.com
	 Date: 24/08/2015
	                          */
	/*** END OF NOTE ***/

	std::vector<int> result, result_count;

	Logger::log(Logger::INFO, " example_sift Launching knn functions...");

	u64 start = getTime();
	GPUGenie::knn_search(result,result_count, config);
	u64 end = getTime();
	double elapsed = getInterval(start, end);

	Logger::log(Logger::DEBUG, ">>>>>>> [time profiling]: Total Time Elapsed: %fms. <<<<<<<", elapsed);

	for(int i = 0; i < 5; ++i)

	{
		printf("Query %d result is: \n\t", i);
		for (int j = 0; j < 10; ++j)
		{
			printf("%d:%d, ", result[i * config.num_of_topk + j], result_count[i * config.num_of_topk + j]);
		}
		printf("\n");
	}

	Logger::exit();

	return 0;
}
