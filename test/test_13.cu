/** Name: test_1.cu
 * Description:
 *   the most basic test
 *   sift data
 *   data is from csv file
 *   query is from csv file, single range
 *
 *
 */


#include "GPUGenie.h"

#include <algorithm>
#include <assert.h>
#include <vector>
#include <iostream>

#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wall"
#include "../fastpfor/headers/codecfactory.h"
#include "../fastpfor/headers/deltautil.h"
#pragma GCC diagnostic po

using namespace std;
using namespace GPUGenie;
using namespace FastPForLib;

int main(int argc, char* argv[])
{

    std::cout << "Available codecs: " << std::endl;
    for (auto &kv : CODECFactory::scodecmap)
        std::cout << "  " << kv.first << std::endl;

    bool delta = true;
    string dataFile = "/home/lubos/data/ocr.dat";
    // string dataFile = "../static/sift_20.dat";
    // string queryFile = "../static/sift_20.csv";
    vector<vector<int> > queries;
    // vector<vector<int> > data;
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

    config.use_load_balance = false;
    config.posting_list_max_length = 6400;
    config.multiplier = 1.5f;
    config.use_multirange = false;

    config.data_type = 1;
    config.search_type = 0;
    config.max_data_size = 0;

    config.num_of_queries = 3;

    assert(config.compression_type == GPUGenie_Config::NO_COMPRESSION);

    std::cout << "Reading data file..." << std::endl;  
    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    // read_file(data, dataFile.c_str(), -1);

    assert(config.item_num > 0);
    assert(config.row_num > 0);

    std::cout << "Done reading data file!" << std::endl;  


    std::cout << "Preprocessing data..." << std::endl;  
    preprocess_for_knn_binary(config, table);
    std::cout << "Done preprocessing data..." << std::endl;  

    // check how many tables we have
    assert(table != NULL);
    assert(table->get_total_num_of_table() == 1);

    std::vector<GPUGenie::inv_list> *inv_lists = table->inv_lists();
    std::cout << "inv_lists.size() (number of attributes): " << inv_lists->size() << std::endl;

    // check inverted index of the tables using inv_list class
    for (int attr_index = 0; attr_index < config.dim; attr_index++)
    {
        GPUGenie::inv_list invertedList = (*inv_lists)[attr_index];
        int posting_list_length = invertedList.size();
        int posting_list_min = invertedList.min();
        int posting_list_max = invertedList.max();
        std::cout << "attr_index " << attr_index << ", posting_list_length:" << posting_list_length
                  << ", min: " << posting_list_min << ", max: " << posting_list_max << std::endl;
        Logger::log(Logger::DEBUG, "attr_index %d, posting_list_length: %d, m in: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        // for (int value = posting_list_min; value <= posting_list_max; ++value)
        // {
        //     vector<int> *docIds = invertedList.index(value);
        //     if (docIds->size())
        //     {
        //         std::stringstream strDocIds;
        //         std::copy(docIds->begin(), docIds->end(), std::ostream_iterator<int>(strDocIds, " "));
        //         Logger::log(Logger::DEBUG, "  value: %d, docIds: %s", value, strDocIds.str().c_str());
        //     }
        // }
    }

    // check what get_lowerbounf_of_list does exactly
    for (int attr_index = 0; attr_index < config.dim; attr_index++)
    {
        Logger::log(Logger::DEBUG, "table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, table->get_lowerbound_of_list(attr_index),
            attr_index, table->get_upperbound_of_list(attr_index));
    }

    // std::stringstream ss;    
    std::vector<int> *ck = table->ck();
    // if (ck)
    // {
    //     auto end = (ck->size() <= 256) ? ck->end() : (ck->begin() + 256); 
    //     std::copy(ck->begin(), end, std::ostream_iterator<int>(ss, " "));
    //     Logger::log(Logger::DEBUG, "CK:\n %s", ss.str().c_str());
    //     ss.str(std::string());
    //     ss.clear();
    // }

    std::vector<int> *inv = table->inv();
    // if (inv)
    // {
    //     auto end = (inv->size() <= 256) ? inv->end() : (inv->begin() + 256); 
    //     std::copy(inv->begin(), end, std::ostream_iterator<int>(ss, " "));
    //     Logger::log(Logger::DEBUG, "INV:\n %s", ss.str().c_str());
    //     ss.str(std::string());
    //     ss.clear();
    // }

    std::vector<int> *inv_index = table->inv_index();
    // if (inv_index)
    // {
    //     auto end = (inv_index->size() <= 256) ? inv_index->end() : (inv_index->begin() + 256); 
    //     std::copy(inv_index->begin(), end, std::ostream_iterator<int>(ss, " "));
    //     Logger::log(Logger::DEBUG, "INV_INDEX:\n %s", ss.str().c_str());
    //     ss.str(std::string());
    //     ss.clear();
    // }


    std::vector<int> *inv_pos = table->inv_pos();
    // if (inv_pos)
    // {
    //     auto end = (inv_pos->size() <= 256) ? inv_pos->end() : (inv_pos->begin() + 256); 
    //     std::copy(inv_pos->begin(), end, std::ostream_iterator<int>(ss, " "));
    //     Logger::log(Logger::DEBUG, "INV_POS:\n %s", ss.str().c_str());
    //     ss.str(std::string());
    //     ss.clear();
    // }

    std::cout << "Copying inverted lists for compression..." << std::endl;
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

        // Debug printout
        // std::copy(invList.begin(), invList.end(), std::ostream_iterator<int>(ss, " "));
        // Logger::log(Logger::DEBUG, "  rawInvertedList:\n %s", ss.str().c_str());
        // ss.str(std::string());
        // ss.clear();
    }
    std::cout << "Done copying inverted lists for compression!" << std::endl;
    
    double avg_inv_list_length = ((double)rawInvertedListsSize) / ((double)inv_pos->size());
    Logger::log(Logger::DEBUG, "Total inverted lists: %d, Average length of inv list: %f",
        rawInvertedListsSize, avg_inv_list_length);

    Logger::log(Logger::DEBUG, "Uncompressed size of inv (bytes): %d", inv->size() * 4);

    Logger::log(Logger::DEBUG, "Uncompressed size of inv_pos (bytes): %d", inv_pos->size() * 4);

    if (delta)
    {
        std::cout << "Delting inverted lists..." << std::endl;
        for (auto it = rawInvertedLists.begin(); it != rawInvertedLists.end(); it++)
            Delta::deltaSIMD(it->data(), it->size());
        std::cout << "Done delting inverted lists..." << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;

    for (auto &kv : CODECFactory::scodecmap)
    {
        string compression_name = kv.first;

        std::cout << "Compressing inverted lists using " << compression_name << "..." << std::endl;
        IntegerCODEC &codec = *CODECFactory::getFromName(compression_name);
        // IntegerCODEC &codec = *CODECFactory::getFromName("simple8b");
        size_t compressedsize_total = 0;
        std::vector<uint32_t> compressed_output;
        for (auto it = rawInvertedLists.begin(); it != rawInvertedLists.end(); it++)
        {
            compressed_output.resize(it->size() + 1024);
            size_t compressedsize = compressed_output.size();
            codec.encodeArray(it->data(), it->size(), compressed_output.data(),compressedsize);
            compressedsize_total += compressedsize;
            // std::cout << "  orig size: " << it->size() << ", compressed size: " << compressedsize << std::endl;
        }
        std::cout << "Done compressing inverted lists..." << std::endl;
        std::cout << "----------------------------------" << std::endl;
        std::cout << std::setprecision(3);
        std::cout << "Ratio: "
                  << 32.0 * static_cast<double>(compressedsize_total) / static_cast<double>(rawInvertedListsSize)
                  << " bits per integer. " << std::endl;
        std::cout << "Compression type: " << compression_name << std::endl;
        std::cout << "Delta: " << (delta ? "SIMD" : "no") << std::endl;
        std::cout << "Data file: " << dataFile << std::endl;
        std::cout << std::endl;
    }
    //
    // You are done!... with the compression...
    //

    // ///
    // // decompressing is also easy:
    // //
    // std::vector<uint32_t> mydataback(N);
    // size_t recoveredsize = mydataback.size();
    // //
    // codec.decodeArray(compressed_output.data(), compressed_output.size(),
    //             mydataback.data(), recoveredsize);
    // mydataback.resize(recoveredsize);
    // //
    // // That's it!
    // //
    // if (mydataback != mydata)
    // throw std::runtime_error("bug!");

    // // If you need to use differential coding, you can use
    // // calls like these to get the deltas and recover the original
    // // data from the deltas:
    // Delta::deltaSIMD(mydata.data(), mydata.size());
    // Delta::inverseDeltaSIMD(mydata.data(), mydata.size());
    // be mindful of CPU caching issues

    return 0;
}

