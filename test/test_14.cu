/**
 * Name: test_14.cu
 * Description:
 *  Test counting of queries on CPU. Similar to CPU-Idx in GENIE paper.
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

const size_t MAX_PRINT_LEN = 128;
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
    string queryFile = DEFAULT_QUERY_DATASET;

    vector<vector<int>> queryPoints;
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

    config.query_points = &queryPoints;
    config.data_points = NULL;

    config.use_load_balance = false;
    config.posting_list_max_length = 6400;
    config.multiplier = 1.5f;
    config.use_multirange = false;

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
    // assert(config.compression_type == GPUGenie_Config::DELTA);
    std::cout << "Done preprocessing data..." << std::endl; 


    std::cout << "Examining inverted lists...";
    std::vector<GPUGenie::inv_list> *inv_lists = table->inv_lists();
    // check inverted index of the tables using inv_list class
    for (size_t attr_index = 0; attr_index < inv_lists->size(); attr_index++)
    {
        GPUGenie::inv_list invertedList = (*inv_lists)[attr_index];
        int posting_list_length = invertedList.size();
        int posting_list_min = invertedList.min();
        int posting_list_max = invertedList.max();
        Logger::log(Logger::DEBUG, "  attr_index %d, posting_list_length: %d, m in: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        Logger::log(Logger::DEBUG, "    table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, table->get_lowerbound_of_list(attr_index),
            attr_index, table->get_upperbound_of_list(attr_index));
    }
    std::cout << "Done examining inverted lists..." << std::endl;

    std::cout << "Copying inverted lists for compression..." << std::endl;

    std::vector<int> *ck = table->ck();
    std::vector<int> *inv = table->inv();
    std::vector<int> *inv_index = table->inv_index();
    std::vector<int> *inv_pos = table->inv_pos();

    std::vector<std::vector<uint32_t>> rawInvertedLists;
    size_t rawInvertedListsSize = inv_pos->back();
    auto inv_it = inv->begin();
    size_t prev_inv_pos = *(inv_pos->begin());
    for (auto inv_pos_it = (inv_pos->begin()+1); inv_pos_it != inv_pos->end(); inv_pos_it++)
    {
        size_t offset = (*inv_pos_it) - prev_inv_pos;
        prev_inv_pos = (*inv_pos_it);
        
        std::vector<uint32_t> invList(inv_it, inv_it + offset);
        inv_it += offset;
        rawInvertedLists.push_back(invList);
    }
    std::cout << "Done copying inverted lists for compression!" << std::endl;
    
    double avg_inv_list_length = ((double)rawInvertedListsSize) / ((double)inv_pos->size());
    Logger::log(Logger::DEBUG, "Total inverted lists: %d, Average length of inv list: %f",
        rawInvertedListsSize, avg_inv_list_length);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv: %d bytes", inv->size() * 4);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv_pos: %d bytes", inv_pos->size() * 4);

    std::cout << std::endl;
    std::cout << std::endl;



    std::cout << "Compressing inverted lists..." << std::endl;
    // for (auto &kv : CODECFactory::scodecmap)
    // {
    // string compression_name = "copy";
    string compression_name = "s4-bp128-d1";
    bool manualDelta = false;

    if (compression_name == "for" || compression_name == "frameofreference"
            || compression_name == "simdframeofreference")
        manualDelta = true;

    // std::cout << "Compressing inverted lists using " << compression_name << "..." << std::endl;
    IntegerCODEC &codec = *CODECFactory::getFromName(compression_name);
    
    size_t compressedsize_total = 0;

    std::vector<std::vector<uint32_t>> comprInvertedLists(rawInvertedLists.size());

    // Compress all inverted lists
    for (size_t i = 0; i < rawInvertedLists.size(); i++)
    {
        comprInvertedLists[i].resize(rawInvertedLists[i].size() + 1024);
        size_t compressedsize = comprInvertedLists[i].size();

        if (manualDelta)
            delta<uint32_t>(static_cast<uint32_t>(0), rawInvertedLists[i].data(), rawInvertedLists[i].size());
        codec.encodeArray(
                rawInvertedLists[i].data(), rawInvertedLists[i].size(),
                comprInvertedLists[i].data(),compressedsize);

        comprInvertedLists[i].resize(compressedsize);
        compressedsize_total += compressedsize;
    }

    std::cout << "Done compressing inverted lists..." << std::endl;


    std::cout << "Preprocessing queries..." << std::endl;

    read_file(*config.query_points, queryFile.c_str(), config.num_of_queries);

    std::vector<query> queries;
    std::vector<int> result;
    std::vector<int> result_count;

    load_query(*table, queries, config);

    for (query &q : queries)
    {
        q.print(MAX_PRINT_LEN);
    }

    knn_search(*table, queries, result, result_count, config);

    {   
        std::stringstream ss;
        auto end = (result_count.size() <= MAX_PRINT_LEN) ? result_count.end() : (result_count.begin()+MAX_PRINT_LEN);
        std::copy(result_count.begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "Results count: %s", ss.str().c_str());
    }
    
    {
        std::stringstream ss;
        auto end = (result.size() <= MAX_PRINT_LEN) ? result.end() : (result.begin() + MAX_PRINT_LEN); 
        std::copy(result.begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "Results: %s", ss.str().c_str());
    }

    // // Decompress all inverted lists
    // unsigned long long time_decompr_start = getTime(), time_decompr_tight_start, time_decompr_tight_stop;
    // double time_decompr_tight = 0.0;
    // for (size_t i = 0; i < rawInvertedLists.size(); i++)
    // {
    //     size_t decompressedsize = rawInvertedLists[i].size();

    //     time_decompr_tight_start = getTime();
    //     codec.decodeArray(
    //         comprInvertedLists[i].data(), comprInvertedLists[i].size(),
    //         rawInvertedLists[i].data(),decompressedsize);
    //     if (manualDelta)
    //         inverseDelta<uint32_t>(static_cast<uint32_t>(0), rawInvertedLists[i].data(),
    //                 rawInvertedLists[i].size());
    //     time_decompr_tight_stop = getTime();

    //     assert(decompressedsize == inv_lists_orig_sizes[i]);
    //     time_decompr_tight += getInterval(time_decompr_tight_start, time_decompr_tight_stop);
    // }
    // unsigned long long time_decompr_stop = getTime();
    // double time_decompr = getInterval(time_decompr_start, time_decompr_stop);

    // std::cout << std::fixed << std::setprecision(3);
    // std::cout << "File: " << dataFile
    //           << ", Compr: " << compression_name
    //           << ", Ratio: "
    //                 << 32.0 * static_cast<double>(compressedsize_total) / static_cast<double>(rawInvertedListsSize)
    //                 << " bpi "
    //           << ", DTime: " << time_decompr
    //           << ", DXTime: " << time_decompr_tight
    //           << std::endl;
    // // }
    // std::cout << "DONE compressing and decompressing inverted lists..." << std::endl;
    // return 0;


    // take a query 

    // uncompress one block of all relevant inverted lists

    // do counting (naively)


    return 0;
}

