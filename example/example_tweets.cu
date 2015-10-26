/*
 * example_tweets.cu
 *
 *  Created on: Oct 16, 2015
 *      Author: zhoujingbo
 *
 * description: This program is to demonstrate the search on string-like data by the GPU. More description of the parameter configuration please refer to example.cu file
 */

#include "../src/GaLG.h" //for ide: change from <GaLG.h> to "../src/GaLG.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>

using namespace GaLG;
using namespace std;


int main4(int argc, char * argv[])//for ide
{
	std::vector<std::vector<int> > queries;
	std::vector<std::vector<int> > data;
	inv_table table;
	read_file(data, "tweets_4k.csv", -1);
	read_file(queries, "tweets_4k.csv", 100);
	GaLG::GaLG_Config config;

	//Data dimension
	//For string search, we use one-dimension-mulitple-values method,
	// i.e. there is only one dimension, all words are considered as the values. 
	//given a query, there can be multiple values for this dimension. 
	//This is like a bag-of-word model for string search
	config.dim = 1;


	config.count_threshold = 0;
	config.hashtable_size = 1.0f;
	config.num_of_topk = 100;
	config.query_radius = 0;
	config.use_device = 0;
	config.num_of_hot_dims = 0;
	config.hot_dim_threshold = 0;

	//use_adaptive_range is not suitable for string search
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;
	config.data_points = &data;
	config.query_points = &queries;
	std::vector<int> result;
	printf("Launching knn functions...\n");
	GaLG::knn_search_tweets(result, config);
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


