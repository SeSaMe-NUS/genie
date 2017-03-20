/** Name: test_11.cu
 * Description:
 *   the most basic test for subsequence search
 *   sift data
 *   data is from csv file
 *   query is from csv file, single range
 *
 *
 */


#include "GPUGenie.h"

#include <assert.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
using namespace GPUGenie;

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    string dataFile = "./sift_1k.dat";
    string queryFile = "./sift_1k.csv";
    vector<vector<int> > queries;
    vector<vector<int> > data;
    inv_table * table = NULL;
    GPUGenie_Config config;

    config.dim = 128;
    config.count_threshold = 256;
    config.num_of_topk = 20;
    config.hashtable_size = 128*config.num_of_topk*1.5;
    config.query_radius = 0;
    config.use_device = 0;
    config.use_adaptive_range = false;
    config.selectivity = 0.0f;

    config.query_points = &queries;
    config.data_points = &data;

    config.use_load_balance = false;
    config.posting_list_max_length = 6400;
    config.multiplier = 1.5f;
    config.use_multirange = false;

    config.data_type = 1;
    config.search_type = 1;
    config.max_data_size = 0;

    config.num_of_queries = 20;
    config.use_subsequence_search = true;
    
    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    //read_file(data, "sift_1k.csv", -1);
    read_file(queries, queryFile.c_str(), config.num_of_queries);




	init_genie(config);
    preprocess_for_knn_binary(config, table);
   

    vector<int> & inv = *table[0].inv();
    


    vector<int> original_result;
    vector<int> result_count;
    vector<int> rowID;
    vector<int> rowOffset;
    knn_search_after_preprocess(config, table, original_result, result_count);

    int shift_bits = table[0]._shift_bits_subsequence();

    get_rowID_offset(original_result, rowID, rowOffset, shift_bits);
    
    
	for(int i = 0; i < config.num_of_queries & i < 5; ++i)
	{
		printf("Query %d result is: \n\t", i);
		for (int j = 0; j < config.num_of_topk && j < 10; ++j)
		{
			printf("%d:%d:%d, ", rowID[i * config.num_of_topk + j], rowOffset[i*config.num_of_topk+j],result_count[i * config.num_of_topk + j]);
		}
		printf("\n");
	}
    reset_device();
    delete[] table;
    return 0;
}
