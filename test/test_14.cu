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

const std::string DEFAULT_TEST_DATASET = "../static/sift_20.dat";
const std::string DEFAULT_QUERY_DATASET = "../static/sift_20.csv";

/**
 *  Sorts GENIE top-k results for each query independently. The top-k results returned from GENIE are in random order,
 *  and if (top-k > number of resutls with match count greater than 0), then remaining docIds in the result vector are
 *  set to 0, thus the result and count vectors cannot be soreted conventionally. 
 */
void sortGenieResults(GPUGenie::GPUGenie_Config &config, std::vector<int> &gpuResultIdxs,
                            std::vector<int> &gpuResultCounts)
{
    std::vector<int> gpuResultHelper(config.num_of_topk),
                     gpuResultHelperTmp(config.num_of_topk);
    for (int queryIndex = 0; queryIndex < config.num_of_queries; queryIndex++)
    {
        int offsetBegin = queryIndex*config.num_of_topk;
        int offsetEnd = (queryIndex+1)*config.num_of_topk;
        // Fint first zero element
        auto firstZeroIt = std::find(gpuResultCounts.begin()+offsetBegin, gpuResultCounts.begin()+offsetEnd, 0);
        // Only sort elements that have non-zero count. This is because GENIE does not return indexed of elements with
        // zero count
        offsetEnd = std::min(offsetEnd,static_cast<int>(
                                    std::distance(gpuResultCounts.begin(),firstZeroIt)));
        
        // Create helper index from 0 to offsetEnd-offsetBegin
        gpuResultHelper.resize(offsetEnd-offsetBegin);
        gpuResultHelperTmp.resize(offsetEnd-offsetBegin);
        std::iota(gpuResultHelper.begin(), gpuResultHelper.end(),0);

        // Sort the helper index according to gpuResultCounts[...+offsetBegin]
        std::sort(gpuResultHelper.begin(),
                  gpuResultHelper.end(),
                  [&gpuResultCounts,offsetBegin](int lhs, int rhs){
                        return (gpuResultCounts[lhs+offsetBegin] > gpuResultCounts[rhs+offsetBegin]);
                    });

        // Shuffle the gpuResultIdxs according to gpuResultHelper
        for (size_t i = 0; i < gpuResultHelper.size(); i++)
            gpuResultHelperTmp[i] = gpuResultIdxs[gpuResultHelper[i]+offsetBegin];
        // Copy back into gpuResultIndex
        std::copy(gpuResultHelperTmp.begin(), gpuResultHelperTmp.end(), gpuResultIdxs.begin()+offsetBegin);

        // Shuffle the gpuResultCounts according to gpuResultHelper
        for (size_t i = 0; i < gpuResultHelper.size(); i++)
            gpuResultHelperTmp[i] = gpuResultCounts[gpuResultHelper[i]+offsetBegin];
        // Copy back into gpuResultIndex
        std::copy(gpuResultHelperTmp.begin(), gpuResultHelperTmp.end(), gpuResultCounts.begin()+offsetBegin); 
    }
}


void getRawInvListSizes(inv_table &table, std::vector<size_t> &rawInvertedListsSizes)
{
    assert(table.build_status() == GPUGenie::inv_table::status::builded);
    std::vector<int> *inv_pos = table.inv_pos();

    rawInvertedListsSizes.clear();
    rawInvertedListsSizes.reserve(inv_pos->size()-1);

    size_t prev_inv_pos = *(inv_pos->begin());
    assert(prev_inv_pos == 0);
    for (auto inv_pos_it = (inv_pos->begin()+1); inv_pos_it != inv_pos->end(); inv_pos_it++)
    {
        size_t sizeOfInvList = (*inv_pos_it) - prev_inv_pos;
        rawInvertedListsSizes.push_back(sizeOfInvList);
        prev_inv_pos = (*inv_pos_it);
    }
}

void compressInvertedLists(
            std::vector<std::vector<uint32_t>> &rawInvertedLists,
            std::vector<std::vector<uint32_t>> &comprInvertedLists,
            const std::string &compression_name,
            bool manualDelta)
{
    IntegerCODEC &codec = *CODECFactory::getFromName(compression_name);
    
    size_t compressedsize_total = 0;

    comprInvertedLists.resize(rawInvertedLists.size());

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
}

void knn_search_cpu(
            inv_table &table,
            std::vector<std::vector<uint32_t>> &comprInvertedLists, // TODO make part of compressed inv_table
            std::vector<query> &queries,
            std::vector<int> &resultIdxs,
            std::vector<int> &resultCounts,
            GPUGenie_Config &config,
            const std::string &compression_name, // TODO make string name of compression part of config 
            bool manualDelta)
{
    assert(config.row_num > 0);
    assert((int)config.row_num >= config.num_of_topk);
    assert(table.build_status() == GPUGenie::inv_table::status::builded);

    std::vector<int> tmpResultIdxs(config.row_num);
    std::vector<int> tmpResultCounts(config.row_num);
    resultCounts.clear();
    resultIdxs.clear();
    resultCounts.reserve(config.num_of_topk * config.num_of_queries);
    resultIdxs.reserve(config.num_of_topk * config.num_of_queries);

    std::vector<size_t> rawInvertedListsSizes;
    getRawInvListSizes(table, rawInvertedListsSizes);
    assert(rawInvertedListsSizes.size());

    unsigned long long time_overall_start, time_overall_stop;
    unsigned long long time_decompr_start, time_decompr_stop;
    unsigned long long time_counting_start, time_counting_stop;
    unsigned long long time_queryPreprocessing_start, time_queryPreprocessing_stop;
    double time_overall = 0, time_decompr = 0, time_counting = 0, time_queryPreprocessing = 0;

    std::vector<std::vector<uint32_t>> rawInvertedLists(rawInvertedListsSizes.size());

    int shifter = table.shifter();
    std::vector<int> *inv_index = table.inv_index();

    IntegerCODEC &codec = *CODECFactory::getFromName(compression_name);

    time_overall_start = getTime();

    for (query &q : queries)
    {
        std::vector<int> invListsTocount;
        std::vector<query::range> ranges;
        int queryIndex = q.index();

        q.dump(ranges);
        Logger::log(Logger::DEBUG, "Processing query %d, has %d ranges", queryIndex, ranges.size());

        if (ranges.empty())
        {
            Logger::log(Logger::ALERT, "Query %d has no ranges!", queryIndex);
            continue;
        }

        time_queryPreprocessing_start = getTime();

        for (query::range &r : ranges)
        {
            int low = r.low;
            int up = r.up;

            int dimShifted = r.dim << shifter;
            
            Logger::log(Logger::DEBUG, "  range %d orig -- query: %d, dim: %d, low: %d, up: %d", r.order, r.query, 
                r.dim, r.low, r.up);

            if(low > up || low > table.get_upperbound_of_list(r.dim) || up < table.get_lowerbound_of_list(r.dim))
            {
                Logger::log(Logger::DEBUG, "  range %d out of bounds of inverted lists in dim %d", r.order, r.dim); 
                continue;
            }

            low = low < table.get_lowerbound_of_list(r.dim) ? table.get_lowerbound_of_list(r.dim) : low;
            up = up > table.get_upperbound_of_list(r.dim) ? table.get_upperbound_of_list(r.dim) : up;

            int min = dimShifted + low - table.get_lowerbound_of_list(r.dim);
            int max = dimShifted + up - table.get_lowerbound_of_list(r.dim);
            Logger::log(Logger::DEBUG, "     processed -- query: %d, dim: %d, low: %d, up: %d, min: %d, max: %d",
                     r.query, r.dim, low, up, min, max);

            // Record ids of inverted lists to be counted
            int invList = (*inv_index)[min];
            do
                invListsTocount.push_back(invList++);
            while (invList < (*inv_index)[max+1]);
        }

        time_queryPreprocessing_stop = getTime();
        time_queryPreprocessing += getInterval(time_queryPreprocessing_start, time_queryPreprocessing_stop);

        // Reset temporary count and index vector -- these vectors are used directly for counting
        std::fill(tmpResultCounts.begin(),tmpResultCounts.end(),0);
        std::iota(tmpResultIdxs.begin(), tmpResultIdxs.end(),0);

        for (int invListIndex : invListsTocount)
        {
            // Decompress the list if we are using compressed table and if it has not been decompressed already
            // if (config.compression_type && rawInvertedLists[invListIndex].size() == 0){
            if (rawInvertedLists[invListIndex].size() == 0){

                Logger::log(Logger::DEBUG, "  decompressing inverted list: %d", invListIndex);
                time_decompr_start = getTime();

                // Get the decompressed size
                // TODO: this should be integral part of the compressed table interface
                size_t decompressedsize = rawInvertedListsSizes[invListIndex];
                // Allocate enough space for the decompressed lists
                rawInvertedLists[invListIndex].resize(decompressedsize);

                // Decompress the compressed inverted list with index invListIndex
                codec.decodeArray(
                    comprInvertedLists[invListIndex].data(), comprInvertedLists[invListIndex].size(),
                    rawInvertedLists[invListIndex].data(),decompressedsize);
                rawInvertedLists[invListIndex].resize(decompressedsize);

                if (manualDelta)
                    inverseDelta<uint32_t>(static_cast<uint32_t>(0), rawInvertedLists[invListIndex].data(),
                            rawInvertedLists[invListIndex].size());

                time_decompr_stop = getTime();
                time_decompr += getInterval(time_decompr_start,time_decompr_stop);

                assert(rawInvertedLists[invListIndex].size() == decompressedsize);
                assert(rawInvertedLists[invListIndex].size() == rawInvertedListsSizes[invListIndex]);
            }

            time_counting_start = getTime();

            // Count docId from the decompressed list
            for (int docId : rawInvertedLists[invListIndex])
                ++tmpResultCounts[docId];

            time_counting_stop = getTime();
            time_counting += getInterval(time_counting_start, time_counting_stop);
        }

        // Sort tmpResultIdxs according to tmpResultCount
        std::sort(tmpResultIdxs.begin(), tmpResultIdxs.end(),
           [&tmpResultCounts](int lhs, int rhs) {return tmpResultCounts[lhs] > tmpResultCounts[rhs];});

        // Copy the first q.topk() results into the final results vectors resultCounts and resultIdxs
        for (auto it = tmpResultIdxs.begin(); it < tmpResultIdxs.begin() + q.topk(); it++)
        {
            resultCounts.push_back(tmpResultCounts[*it]);
            resultIdxs.push_back(*it);
        }
    }

    time_overall_stop = getTime();
    time_overall = getInterval(time_overall_start, time_overall_stop);

    std::cout << std::fixed << std::setprecision(3)
              << "time_overall: " << time_overall
              << ", time_decompr: " << time_decompr
              << ", time_queryPreprocessing: " << time_queryPreprocessing
              << ", time_counting: " << time_counting
              << std::endl;
}


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
    config.num_of_topk = 20;
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
        Logger::log(Logger::DEBUG, "  attr_index %d, posting_list_length: %d, min: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        Logger::log(Logger::DEBUG, "    table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, table->get_lowerbound_of_list(attr_index),
            attr_index, table->get_upperbound_of_list(attr_index));
    }

    Logger::logTable(Logger::DEBUG,table);

    std::cout << "Done examining inverted lists..." << std::endl;

    std::cout << "Extracting inverted lists for compression..." << std::endl;

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

    Logger::logInvLists(Logger::DEBUG,rawInvertedLists);

    std::cout << "Done extracting inverted lists for compression!" << std::endl;
    
    double avg_inv_list_length = ((double)rawInvertedListsSize) / ((double)inv_pos->size());
    Logger::log(Logger::DEBUG, "Total inverted lists: %d, Average length of inv list: %f",
        rawInvertedListsSize, avg_inv_list_length);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv: %d bytes", inv->size() * 4);
    Logger::log(Logger::DEBUG, "Uncompressed size of inv_pos: %d bytes", inv_pos->size() * 4);


    std::cout << "Loading queries..." << std::endl;

    read_file(*config.query_points, queryFile.c_str(), config.num_of_queries);
    std::vector<query> queries;
    load_query(*table, queries, config);

    std::cout << "Done loading queries..." << std::endl;


    std::cout << "Running KNN on GPU..." << std::endl;

    std::vector<int> gpuResultIdxs;
    std::vector<int> gpuResultCounts;
    knn_search(*table, queries, gpuResultIdxs, gpuResultCounts, config);

    // Top k results from GENIE don't have to be sorted. In order to compare with CPU implementation, we have to
    // sort the results manually from individual queries => sort subsequence relevant to each query independently
    sortGenieResults(config, gpuResultIdxs, gpuResultCounts);

    Logger::log(Logger::DEBUG, "Results from GENIE:");
    Logger::logResults(Logger::DEBUG, queries, gpuResultIdxs, gpuResultCounts);

    std::cout << "Done running KNN on GPU..." << std::endl;



    for (auto &kv : CODECFactory::scodecmap)
    {
        string compression_name = kv.first;
        bool manualDelta = false;
        if (compression_name == "for" || compression_name == "frameofreference"
                || compression_name == "simdframeofreference")
            manualDelta = true;


        std::vector<std::vector<uint32_t>> rawInvertedListsCopy(rawInvertedLists);
        std::vector<std::vector<uint32_t>> comprInvertedLists;

        std::cout << "Compressing inverted lists with " << compression_name << std::endl;
        compressInvertedLists(rawInvertedListsCopy, comprInvertedLists, compression_name, manualDelta);
        std::cout << "Done compressing inverted lists..." << std::endl;


        std::vector<int> resultIdxs;
        std::vector<int> resultCounts;

        std::cout << "Running KNN on CPU..." << std::endl;
        std::cout << "KNN_SEARCH_CPU"
                  << ", file: " << dataFile << " (" << config.row_num << " rows)" 
                  << ", queryFile: " << queryFile << " (" << config.num_of_queries << " queries)"
                  << ", topk: " << config.num_of_topk
                  << ", compression: " << compression_name
                  << ", ";
        knn_search_cpu(*table, comprInvertedLists, queries, resultIdxs, resultCounts, config, compression_name,
                manualDelta);
        Logger::log(Logger::DEBUG, "Results from CPU naive decompressed counting:");
        Logger::logResults(Logger::DEBUG, queries, resultIdxs, resultCounts);
        std::cout << "Done running KNN on CPU..." << std::endl;

        // Compare the first docId from the GPU and CPU results -- note since we use points from the data file
        // as queries, One of the resutls is a full-dim count match (self match), which is what we compare here.
        assert(gpuResultIdxs[0 * config.num_of_topk] == resultIdxs[0 * config.num_of_topk]
            && gpuResultCounts[0 * config.num_of_topk] == resultCounts[0 * config.num_of_topk]);
        assert(gpuResultIdxs[1 * config.num_of_topk] == resultIdxs[1 * config.num_of_topk]
            && gpuResultCounts[1 * config.num_of_topk] == resultCounts[1 * config.num_of_topk]);
        assert(gpuResultIdxs[2 * config.num_of_topk] == resultIdxs[2 * config.num_of_topk]
            && gpuResultCounts[2 * config.num_of_topk] == resultCounts[2 * config.num_of_topk]);
        // if((gpuResultIdxs[0 * config.num_of_topk] == resultIdxs[0 * config.num_of_topk]
        //         && gpuResultCounts[0 * config.num_of_topk] == resultCounts[0 * config.num_of_topk]) 
        //         &&(gpuResultIdxs[1 * config.num_of_topk] == resultIdxs[1 * config.num_of_topk]
        //         && gpuResultCounts[1 * config.num_of_topk] == resultCounts[1 * config.num_of_topk])
        //         &&(gpuResultIdxs[2 * config.num_of_topk] == resultIdxs[2 * config.num_of_topk]
        //         && gpuResultCounts[2 * config.num_of_topk] == resultCounts[2 * config.num_of_topk]))
        //     std::cout << "Compression " << compression_name << " succeeded!" << std::endl;
        // else
        //     std::cout << "Compression " << compression_name << " FAILED!" << std::endl;
    }

    return 0;
}

