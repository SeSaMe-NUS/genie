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

int main(int argc, char* argv[])
{
    Logger::log(Logger::INFO, "Available compressions in GENIE (GPUGenie_Config::COMPRESSION_NAMES):");
    for (std::string &compr : GPUGenie_Config::COMPRESSION_NAMES)
        Logger::log(Logger::INFO, "  %s", compr.c_str());


    string dataFile = DEFAULT_TEST_DATASET;
    if (argc == 2)
        dataFile = std::string(argv[1]);
    string queryFile = DEFAULT_QUERY_DATASET;

    vector<vector<int>> queryPoints;
    GPUGenie_Config config;

    config.dim = 5;
    config.count_threshold = 14;
    config.num_of_topk = 10;
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



    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Establishing reference solution on uncompressed table..." << std::endl;

    std::cout << "Preprocessing data (" << config.item_num << " items total)..." << std::endl;  
    inv_table * refTable = NULL;
    preprocess_for_knn_binary(config, refTable);
    assert(refTable != NULL);
    assert(refTable->get_total_num_of_table() == 1); // check how many tables we have
    assert(dynamic_cast<inv_compr_table*>(refTable) == NULL);

    std::cout << "Examining inverted lists...";
    std::vector<GPUGenie::inv_list> *invLists = refTable->inv_lists();
    // check inverted index of the tables using inv_list class
    for (size_t attr_index = 0; attr_index < invLists->size(); attr_index++)
    {
        GPUGenie::inv_list invertedList = (*invLists)[attr_index];
        int posting_list_length = invertedList.size();
        int posting_list_min = invertedList.min();
        int posting_list_max = invertedList.max();
        Logger::log(Logger::DEBUG, "  attr_index %d, posting_list_length: %d, min: %d, max: %d",
                        attr_index, posting_list_length, posting_list_min, posting_list_max);
        Logger::log(Logger::DEBUG, "    table->get_lowerbound_of_list(%d): %d, table->get_upperbound_of_list(%d): %d", attr_index, refTable->get_lowerbound_of_list(attr_index),
            attr_index, refTable->get_upperbound_of_list(attr_index));
    }

    Logger::logTable(Logger::DEBUG,refTable);

    std::cout << "Loading queries..." << std::endl;
    read_file(*config.query_points, queryFile.c_str(), config.num_of_queries);
    std::vector<query> refQueries;
    load_query(*refTable, refQueries, config);

    std::cout << "Running KNN on GPU..." << std::endl;
    std::vector<int> refResultIdxs;
    std::vector<int> refResultCounts;
    knn_search(*refTable, refQueries, refResultIdxs, refResultCounts, config);
    // Top k results from GENIE don't have to be sorted. In order to compare with CPU implementation, we have to
    // sort the results manually from individual queries => sort subsequence relevant to each query independently
    sortGenieResults(config, refResultIdxs, refResultCounts);
    Logger::log(Logger::DEBUG, "Results from GENIE:");
    Logger::logResults(Logger::DEBUG, refQueries, refResultIdxs, refResultCounts);



    std::cout << "---------------------------" << std::endl;
    std::cout << "Testing compressed table..." << std::endl;

    config.compression = "copy";

    std::cout << "Preprocessing data (" << config.item_num << " items total)..." << std::endl;

    inv_table * table = NULL;
    inv_compr_table * comprTable = NULL;
    preprocess_for_knn_binary(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != NULL);
    assert(table->build_status() == inv_table::builded);
    assert(table->get_total_num_of_table() == 1); // check how many tables we have
    comprTable = dynamic_cast<inv_compr_table*>(table);
    assert(config.posting_list_max_length == (int)comprTable->getUncompressedPostingListMaxLength());
    assert(config.compression == comprTable->getCompression()); // check the compression was actually used in the table

    std::cout << "Examining compressed index..." << std::endl;

    std::vector<int> *uncompressedInv = comprTable->uncompressedInv();
    std::vector<int> *uncompressedInvPos = comprTable->uncompressedInvPos();
    std::vector<uint32_t> *compressedInv = comprTable->compressedInv();
    std::vector<int> *compressedInvPos = comprTable->compressedInvPos();
    // the last elm in inv_pos should be the compressed size, which is <= to the original size
    assert(compressedInvPos->size() == uncompressedInvPos->size());
    assert(compressedInvPos->back() == (int)compressedInv->size()); 
    assert(compressedInvPos->back() <= uncompressedInvPos->back()); // compression should not enlarge data
    assert(compressedInv == reinterpret_cast<std::vector<uint32_t>*>(comprTable->inv())); // test alias function
    assert(compressedInvPos == comprTable->inv_pos()); // test alias function

    double avg_inv_list_length = ((double)uncompressedInv->size()) / ((double)uncompressedInvPos->size());
    double avg_compr_inv_list_length = ((double)compressedInv->size()) / ((double)compressedInvPos->size());
    Logger::log(Logger::DEBUG,
            "Uncompressed inverted list length: %d, Inverted lists: %d, Average length of uncompressed inv list: %f",
            uncompressedInv->size(), uncompressedInvPos->size(), avg_inv_list_length);
    Logger::log(Logger::DEBUG, "Compressed size of posting lists array Z: %d bytes", compressedInv->size() * 4);
    Logger::log(Logger::DEBUG, "Uncompressed size of compressedInvPos index: %d bytes", compressedInvPos->size() * 4);
    Logger::log(Logger::DEBUG, "Average size of compressed posting list: %d", avg_compr_inv_list_length);

    Logger::logTable(Logger::DEBUG,comprTable);


    std::cout << "Loading queries..." << std::endl;
    std::vector<query> queries;
    load_query(*comprTable, queries, config);

    std::vector<int> resultIdxs;
    std::vector<int> resultCounts;

    std::cout << "Running KNN on GPU (compression, naive counting)" << std::endl;
    std::cout << "KNN_SEARCH_CPU"
              << ", file: " << dataFile << " (" << config.row_num << " rows)" 
              << ", queryFile: " << queryFile << " (" << config.num_of_queries << " queries)"
              << ", topk: " << config.num_of_topk
              << ", compression: " << config.compression
              << ", ";

    knn_search(*comprTable, queries, resultIdxs, resultCounts, config);
    sortGenieResults(config, resultIdxs, resultCounts);

    Logger::log(Logger::DEBUG, "Results from GPU naive decompressed counting:");
    Logger::logResults(Logger::DEBUG, queries, resultIdxs, resultCounts);



    std::cout<< "---------------------------------------------" << std::endl;
    std::cout<< "Comparing reference and compressed results..." << std::endl;

    // Compare the first docId from the GPU and CPU results -- note since we use points from the data file
    // as queries, One of the resutls is a full-dim count match (self match), which is what we compare here.
    assert(refResultIdxs[0 * config.num_of_topk] == resultIdxs[0 * config.num_of_topk]
        && refResultCounts[0 * config.num_of_topk] == resultCounts[0 * config.num_of_topk]);
    assert(refResultIdxs[1 * config.num_of_topk] == resultIdxs[1 * config.num_of_topk]
        && refResultCounts[1 * config.num_of_topk] == resultCounts[1 * config.num_of_topk]);
    assert(refResultIdxs[2 * config.num_of_topk] == resultIdxs[2 * config.num_of_topk]
        && refResultCounts[2 * config.num_of_topk] == resultCounts[2 * config.num_of_topk]);

    return 0;
}

