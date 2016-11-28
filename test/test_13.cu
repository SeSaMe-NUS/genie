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

const std::string DEFAULT_TEST_DATASET = "../static/tweets_20.dat";

void log_table(GPUGenie::inv_table *table, size_t max_print_len = 256)
{
    if (table->build_status() == GPUGenie::inv_table::not_builded)
    {
        Logger::log(Logger::DEBUG, "Inv table not built.");
        return;
    }

    std::stringstream ss;    
    std::vector<int> *ck = table->ck();
    if (ck)
    {
        auto end = (ck->size() <= max_print_len) ? ck->end() : (ck->begin() + max_print_len); 
        std::copy(ck->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "CK:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }

    std::vector<int> *inv = table->inv();
    if (inv)
    {
        auto end = (inv->size() <= max_print_len) ? inv->end() : (inv->begin() + max_print_len); 
        std::copy(inv->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }

    std::vector<int> *inv_index = table->inv_index();
    if (inv_index)
    {
        auto end = (inv_index->size() <= max_print_len) ? inv_index->end() : (inv_index->begin() + max_print_len); 
        std::copy(inv_index->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV_INDEX:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }


    std::vector<int> *inv_pos = table->inv_pos();
    if (inv_pos)
    {
        auto end = (inv_pos->size() <= max_print_len) ? inv_pos->end() : (inv_pos->begin() + max_print_len); 
        std::copy(inv_pos->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV_POS:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }
}

void log_inv_lists(const std::vector<std::vector<uint32_t>> &rawInvertedLists, size_t max_print_len = 16)
{
    std::stringstream ss;
    auto inv_it_end = (rawInvertedLists.size() <= max_print_len)
                            ? rawInvertedLists.end() : (rawInvertedLists.begin() + max_print_len);
    Logger::log(Logger::DEBUG, "rawInvertedLists.size(): %d", rawInvertedLists.size());
    for (auto inv_it = rawInvertedLists.begin(); inv_it != inv_it_end; inv_it++)
    {
        const std::vector<uint32_t> &invList = *inv_it; 
        auto end = (invList.size() <= max_print_len) ? invList.end() : (invList.begin() + max_print_len);
        std::copy(invList.begin(), end, std::ostream_iterator<uint32_t>(ss, " "));
        Logger::log(Logger::DEBUG, "*** [%s]", ss.str().c_str());
        ss.str(std::string());
        ss.clear();

    }
}


int main(int argc, char* argv[])
{
    std::cout << "Available codecs (SIMDCompressionLib::CODECFactory::scodecmap):" << std::endl;
    for (auto &kv : CODECFactory::scodecmap)
        std::cout << "  " << kv.first << std::endl;

    
    string dataFile = DEFAULT_TEST_DATASET;
    if (argc == 2)
        dataFile = std::string(argv[1]);

    inv_table * table = NULL;
    GPUGenie_Config config;

    config.data_points = NULL;
    config.use_load_balance = false;
    config.data_type = 1;

    assert(config.compression_type == GPUGenie_Config::NO_COMPRESSION);

    std::cout << "Reading data file " << dataFile << "..." << std::endl;  
    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    assert(config.item_num > 0);
    assert(config.row_num > 0);
    Logger::log(Logger::DEBUG, "config.item_num: %d", config.item_num);
    Logger::log(Logger::DEBUG, "config.row_num: %d", config.row_num);
    std::cout << "Done reading data file!" << std::endl;  


    std::cout << "Preprocessing data (" << config.item_num << " items total)..." << std::endl;  
    preprocess_for_knn_binary(config, table);
    std::cout << "Done preprocessing data..." << std::endl;  

    // check how many tables we have
    assert(table != NULL);
    assert(table->get_total_num_of_table() == 1);

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

    std::vector<int> *ck = table->ck();
    std::vector<int> *inv = table->inv();
    std::vector<int> *inv_index = table->inv_index();
    std::vector<int> *inv_pos = table->inv_pos();

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
    }
    log_inv_lists(rawInvertedLists);
    std::cout << "Done copying inverted lists for compression!" << std::endl;
    
    double avg_inv_list_length = ((double)rawInvertedListsSize) / ((double)inv_pos->size());
    Logger::log(Logger::DEBUG, "Total inverted lists: %d, Average length of inv list: %f",
        rawInvertedListsSize, avg_inv_list_length);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv: %d bytes", inv->size() * 4);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv_pos: %d bytes", inv_pos->size() * 4);

    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Compressing and decompressing inverted lists..." << std::endl;
    for (auto &kv : CODECFactory::scodecmap)
    {
        string compression_name = kv.first;

        // std::cout << "Compressing inverted lists using " << compression_name << "..." << std::endl;
        IntegerCODEC &codec = *CODECFactory::getFromName(compression_name);
        
        size_t compressedsize_total = 0;

        std::vector<std::vector<uint32_t>> compressed_output(rawInvertedLists.size());
        std::vector<size_t> inv_lists_orig_sizes(rawInvertedLists.size());

        // Compress all inverted lists
        unsigned long long time_compr_start = getTime(), time_compr_tight_start, time_compr_tight_stop;
        double time_compr_tight = 0.0;
        for (size_t i = 0; i < rawInvertedLists.size(); i++)
        {
            inv_lists_orig_sizes[i] = rawInvertedLists[i].size();
            compressed_output[i].resize(rawInvertedLists[i].size() + 1024);
            size_t compressedsize = compressed_output[i].size();

            time_compr_tight_start = getTime();
            codec.encodeArray(
                    rawInvertedLists[i].data(), rawInvertedLists[i].size(),
                    compressed_output[i].data(),compressedsize);
            time_compr_tight_stop = getTime();

            compressed_output[i].resize(compressedsize);
            compressedsize_total += compressedsize;
            time_compr_tight += getInterval(time_compr_tight_start, time_compr_tight_stop);
        }
        unsigned long long time_compr_stop = getTime();
        double time_compr = getInterval(time_compr_start, time_compr_stop);

        // Decompress all inverted lists
        unsigned long long time_decompr_start = getTime(), time_decompr_tight_start, time_decompr_tight_stop;
        double time_decompr_tight = 0.0;
        for (size_t i = 0; i < rawInvertedLists.size(); i++)
        {
            size_t decompressedsize = rawInvertedLists[i].size();

            time_decompr_tight_start = getTime();
            codec.decodeArray(
                compressed_output[i].data(), compressed_output[i].size(),
                rawInvertedLists[i].data(),decompressedsize);
            time_decompr_tight_stop = getTime();

            assert(decompressedsize == inv_lists_orig_sizes[i]);
            time_decompr_tight += getInterval(time_decompr_tight_start, time_decompr_tight_stop);
        }
        unsigned long long time_decompr_stop = getTime();
        double time_decompr = getInterval(time_decompr_start, time_decompr_stop);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "File: " << dataFile
                  << ", Compr: " << compression_name
                  << ", Ratio: "
                        << 32.0 * static_cast<double>(compressedsize_total) / static_cast<double>(rawInvertedListsSize)
                        << " bpi "
                  << ", CTime: " << time_compr
                  << ", DTime: " << time_decompr
                  << ", CXTime: " << time_compr_tight
                  << ", DXTime: " << time_decompr_tight
                  << std::endl;
    }
    std::cout << "DONE compressing and decompressing inverted lists..." << std::endl;
    return 0;
}

