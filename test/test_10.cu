/** Name: test_10.cu
 * Description:
 * focus on multiload
 *   tweet data(short text)
 *   data is from binary file
 *   query is from csv file, single range
 *
 *
 */


#undef NDEBUG
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
    config.num_of_topk = 5;
    config.hashtable_size = 100*config.num_of_topk*1.5;
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
    config.max_data_size = 10;

    config.num_of_queries = 3;

    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    read_file(queries, queryFile.c_str(), config.num_of_queries);

	init_genie(config);
    preprocess_for_knn_binary(config, table);

    assert(table[0].get_total_num_of_table() == 2);
    /**test for table*/
    vector<int>& inv = *table[0].inv();
    
    assert(inv[0] == 0);
    assert(inv[1] == 0);
    assert(inv[2] == 0);
    assert(inv[3] == 0);
    assert(inv[4] == 0);
    assert(inv[5] == 7);

    vector<int> result;
    vector<int> result_count;
    knn_search_after_preprocess(config, table, result, result_count);

    assert(result[0] == 0 || result[0] == 11);
    assert(result_count[0] == 16);
    assert(result_count[1] == 16);

    assert(result[5] == 1);
    assert(result_count[5] == 14);
    
    assert(result[10] == 2);
    assert(result_count[10] == 16);
    delete[] table;
    return 0;
}
