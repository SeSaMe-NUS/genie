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

#include "../fastpfor/headers/codecfactory.h"
#include "../fastpfor/headers/deltautil.h"

using namespace std;
using namespace GPUGenie;
using namespace FastPForLib;

int main(int argc, char* argv[])
{

    // We pick a CODEC
 
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
    // could use others, e.g., "simdbinarypacking", "varintg8iu"
    ////////////
    //
    // create a container with some integers in it
    //
    // for this example, we will not assume that the
    // integers are in sorted order
    //
    // (Note: You don't need to use a vector.)
    //
    size_t N = 10 * 1000;
    std::vector<uint32_t> mydata(N);
    for (uint32_t i = 0; i < N; i += 150)
        mydata[i] = i;
    //
    // the vector mydata could contain anything, really
    //
    ///////////
    //
    // You need some "output" container. You are responsible
    // for allocating enough memory.
    //
    std::vector<uint32_t> compressed_output(N + 1024);
    // N+1024 should be plenty
    //
    //
    size_t compressedsize = compressed_output.size();
    codec.encodeArray(mydata.data(), mydata.size(), compressed_output.data(),
                    compressedsize);
    //
    // if desired, shrink back the array:
    compressed_output.resize(compressedsize);
    compressed_output.shrink_to_fit();
    // display compression rate:
    std::cout << std::setprecision(3);
    std::cout << "You are using "
            << 32.0 * static_cast<double>(compressed_output.size()) /
                   static_cast<double>(mydata.size())
            << " bits per integer. " << std::endl;
    //
    // You are done!... with the compression...
    //

    ///
    // decompressing is also easy:
    //
    std::vector<uint32_t> mydataback(N);
    size_t recoveredsize = mydataback.size();
    //
    codec.decodeArray(compressed_output.data(), compressed_output.size(),
                mydataback.data(), recoveredsize);
    mydataback.resize(recoveredsize);
    //
    // That's it!
    //
    if (mydataback != mydata)
    throw std::runtime_error("bug!");

    // If you need to use differential coding, you can use
    // calls like these to get the deltas and recover the original
    // data from the deltas:
    Delta::deltaSIMD(mydata.data(), mydata.size());
    Delta::inverseDeltaSIMD(mydata.data(), mydata.size());
    // be mindful of CPU caching issues

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
    config.max_data_size = 0;

    config.num_of_queries = 3;

    assert(config.compression_type == GPUGenie_Config::NO_COMPRESSION);

    read_file(data, dataFile.c_str(), -1);
    read_file(queries, queryFile.c_str(), config.num_of_queries);

    preprocess_for_knn_csv(config, table);

    // check how many tables we have
    assert(table != NULL);
    assert(table->get_total_num_of_table() == 1);

    std::vector<GPUGenie::inv_list> *inv_lists = table->inv_lists();
    std::cout << "inv_lists.size(): " << inv_lists->size() << std::endl;

    // check inverted index of the tables using inv_list class
    for (int attr_index = 0; attr_index < config.dim; attr_index++)
    {
        GPUGenie::inv_list invertedList = (*inv_lists)[attr_index];
        int posting_list_length = invertedList.size();
        int posting_list_min = invertedList.min();
        int posting_list_max = invertedList.max();
        Logger::log(Logger::DEBUG, "attr_index %d, posting_list_length: %d, min: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        for (int value = posting_list_min; value <= posting_list_max; ++value)
        {
            vector<int> *docIds = invertedList.index(value);
            if (docIds->size())
            {
                std::stringstream strDocIds;
                std::copy(docIds->begin(), docIds->end(), std::ostream_iterator<int>(strDocIds, " "));
                Logger::log(Logger::DEBUG, "  value: %d, docIds: %s", value, strDocIds.str().c_str());
            }
        }
    }

    // check what get_lowerbounf_of_list does exactly
    for (int attr_index = 0; attr_index < config.dim; attr_index++)
    {
        Logger::log(Logger::DEBUG, "table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, table->get_lowerbound_of_list(attr_index),
            attr_index, table->get_upperbound_of_list(attr_index));
    }

    std::stringstream ss;

    std::vector<int> *ck = table->ck();
    if (ck)
    {
        auto end = (ck->size() <= 256) ? ck->end() : (ck->begin() + 256); 
        std::copy(ck->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "CK:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }

    std::vector<int> *inv = table->inv();
    if (inv)
    {
        auto end = (inv->size() <= 256) ? inv->end() : (inv->begin() + 256); 
        std::copy(inv->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }

    std::vector<int> *inv_index = table->inv_index();
    if (inv_index)
    {
        auto end = (inv_index->size() <= 256) ? inv_index->end() : (inv_index->begin() + 256); 
        std::copy(inv_index->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV_INDEX:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }


    std::vector<int> *inv_pos = table->inv_pos();
    if (inv_pos)
    {
        auto end = (inv_pos->size() <= 256) ? inv_pos->end() : (inv_pos->begin() + 256); 
        std::copy(inv_pos->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(Logger::DEBUG, "INV_POS:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }

    // check values / print values into a file
    // is there a function that can be used for that?

    /**test for table*/
    vector<int>& _inv = *table[0].inv();
    assert(_inv[0] == 8);
    assert(_inv[1] == 9);
    assert(_inv[2] == 7);
    assert(_inv[3] == 0);
    assert(_inv[4] == 2);
    assert(_inv[5] == 4);

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
    delete[] table;
    return 0;
}

