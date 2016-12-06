/**
 * Name: test_13.cu
 * Description:
 *  Test for integer compression library lemire/SIMDCompressionAndIntersection and it's integration into GENIE  
 */

#include <GPUGenie.h>

#include <algorithm>
#include <assert.h>
#include <vector>
#include <iostream>

#include <sstream>
#include <stdio.h>

#include "codecfactory.h"
#include "intersection.h"

using namespace GPUGenie;
using namespace SIMDCompressionLib;

const std::string DEFAULT_TEST_DATASET = "../static/sift_20.dat";
const std::string DEFAULT_QUERY_DATASET = "../static/sift_20.csv";

int main(int argc, char* argv[])
{
    Logger::log(Logger::INFO, "Available codecs (SIMDCompressionLib::CODECFactory::scodecmap):");
    for (auto &kv : CODECFactory::scodecmap)
        Logger::log(Logger::INFO, "  %s", kv.first.c_str());
    
    // Logger::log(Logger::INFO, "Available compressions in GENIE (GPUGenie_Config::):");
    // for (auto &kv : GPUGenie_Config::compression_types)
    //     Logger::log(Logger::INFO, "  %s", kv.first);


    string dataFile = DEFAULT_TEST_DATASET;
    if (argc == 2)
        dataFile = std::string(argv[1]);

    vector<vector<int> > queries;
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
    config.data_points = NULL;

    config.data_points = NULL;
    config.use_load_balance = false;

    config.data_type = 1;
    config.search_type = 0;
    config.max_data_size = 0;

    config.num_of_queries = 3;

    config.compression_type = GPUGenie_Config::COMPRESSION_TYPE::NO_COMPRESSION;


    std::cout << "Reading data file " << dataFile << "..." << std::endl;  
    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    assert(config.item_num > 0);
    assert(config.row_num > 0);
    Logger::log(Logger::DEBUG, "config.item_num: %d", config.item_num);
    Logger::log(Logger::DEBUG, "config.row_num: %d", config.row_num);
    std::cout << "Done reading data file!" << std::endl;  


    std::cout << "Preprocessing data (" << config.item_num << " items total)..." << std::endl;  
    preprocess_for_knn_binary(config, table);
    // check how many tables we have
    assert(table != NULL);
    assert(table->get_total_num_of_table() == 1);
    assert(config.compression_type == GPUGenie_Config::DELTA);
    std::cout << "Done preprocessing data..." << std::endl; 

    std::cout << "Running queries on compressed table..." << std::endl;
    vector<int> result;
    vector<int> result_count;
    knn_search_after_preprocess(config, table, result, result_count);
    assert(result[0] == 0);
    assert(result_count[0] == 5);
    assert(result[1] == 4);
    assert(result_count[1] == 2);
    assert(result[5] == 1);
    assert(result_count[5] == 5);
    assert(result[10] == 2);
    assert(result_count[10] == 5);
    std::cout << "Done running queries on compressed table..." << std::endl;


    std::cout << "Examining inverted lists...";
    std::vector<GPUGenie::inv_list> *inv_lists = table->inv_lists();
    // check inverted index of the tables using inv_list class
    for (int attr_index = 0; attr_index < inv_lists->size(); attr_index++)
    {
        GPUGenie::inv_list invertedList = (*inv_lists)[attr_index];
        int posting_list_length = invertedList.size();
        int posting_list_min = invertedList.min();
        int posting_list_max = invertedList.max();
        Logger::log(Logger::DEBUG, "  attr_index %d,\n    posting_list_length: %d, m in: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        Logger::log(Logger::DEBUG, "    table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, table->get_lowerbound_of_list(attr_index),
            attr_index, table->get_upperbound_of_list(attr_index));
    }
    std::cout << "Done examining  inverted lists..." << std::endl;


    return 0;
}

