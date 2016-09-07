/** Name: test_6.cu
 * Description:
 * focus on serialization of table, especially under multiload setting
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
    string dataFile = "../static/sift_20.csv";
    string queryFile = "../static/sift_20.csv";
    vector<vector<int> > queries;
    vector<vector<int> > data;
    inv_table * table = NULL;
    GPUGenie_Config config;

    config.dim = 5;
    config.count_threshold = 14;
    config.num_of_topk = 5;
    config.hashtable_size = 14*config.num_of_topk*1.5;
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

    config.data_type = 0;
    config.search_type = 0;
    config.max_data_size = 10;

    config.num_of_queries = 3;

    read_file(data, dataFile.c_str(), -1);
    read_file(queries, queryFile.c_str(), config.num_of_queries);

    unsigned int table_num;
    preprocess_for_knn_csv(config, table);

    table_num = table[0].get_total_num_of_table();
    assert(table_num == 2);
    assert(inv_table::write("test_table_binaryfile.dat", table));
    delete[] table;

    inv_table* _table;
    assert(inv_table::read("test_table_binaryfile.dat", _table));

    

    table_num = _table[0].get_total_num_of_table();
    assert(table_num == 2);
    
    vector<int>& inv = *_table[0].inv();
    assert(inv[0] == 8);
    assert(inv[1] == 9);
    assert(inv[2] == 7);
    assert(inv[3] == 0);
    assert(inv[4] == 2);
    assert(inv[5] == 4);

    vector<int> result;
    vector<int> result_count;
    knn_search_after_preprocess(config, _table, result, result_count);

    assert(result[0] == 0);
    assert(result_count[0] == 5);

    assert(result[1] == 4);
    assert(result_count[1] == 2);

    assert(result[5] == 1);
    assert(result_count[5] == 5);
    
    assert(result[10] == 2);
    assert(result_count[10] == 5);
    delete[] _table;
    return 0;
}
