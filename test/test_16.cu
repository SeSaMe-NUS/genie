/** Name: test_16.cu
 * Description:
 *      Tests for hash table overflow handling in matching kernel
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
    string dataFile = "../static/tweets_20.dat";
    string queryFile = "../static/tweets_20.csv";
    vector<vector<int> > queries;
    vector<vector<int> > data;
    inv_table * table = NULL;
    GPUGenie_Config config;

    config.dim = 14;
    config.count_threshold = 14;
    config.num_of_topk = 10;

    // Intentionally make hashtable_size very small
    // Matching kernel has to run several times with gradually increased table size 
    // GENIE resets hash table size if config.hashtable_size <= 2
    config.hashtable_size = 3;
    config.query_radius = 0;
    config.use_device = 0;
    config.use_adaptive_range = false;
    config.selectivity = 0.0f;

    config.query_points = &queries;
    config.data_points = NULL;

    config.use_load_balance = false;
    config.posting_list_max_length = 6400;
    config.multiplier = 1.5f;
    config.use_multirange = false;

    config.data_type = 1;
    config.search_type = 1;
    config.max_data_size = 0;

    config.num_of_queries = 20;

    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    read_file(queries, queryFile.c_str(), config.num_of_queries);

	init_genie(config);
    preprocess_for_knn_binary(config, table);

    vector<int> result;
    vector<int> result_count;
    knn_search_after_preprocess(config, table, result, result_count);

    delete[] table;
    return 0;
}

