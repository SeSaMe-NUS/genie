/**
 * Name: test_18.cu
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
const int         DEFAULT_DIMENSIONS = 5;
const int         DEFAULT_NUM_QUERIES = 3;


/**
 *  Sorts GENIE top-k results for each query independently. The top-k results returned from GENIE are in random order,
 *  and if (top-k > number of resutls with match count greater than 0), then remaining docIds in the result vector are
 *  set to 0, thus the result and count vectors cannot be sorted conventionally. 
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

void checkDataFileIsNotBinary(const std::string &dataFile)
{
    Logger::log(Logger::INFO, "Checking if dataFile argument is not .inv or .cinv file", dataFile.c_str());

    std::string invSuffix(".inv");
    std::string cinvSuffix(".cinv");

    if (dataFile.size() >= invSuffix.size() + 1
            && std::equal(invSuffix.rbegin(), invSuffix.rend(), dataFile.rbegin())){
        Logger::log(Logger::ALERT, "dataFile %s is an inv_table binary file", dataFile.c_str());
        exit(1);
    }
    if (dataFile.size() >= invSuffix.size() + 1
            && std::equal(cinvSuffix.rbegin(), cinvSuffix.rend(), dataFile.rbegin())){
        Logger::log(Logger::ALERT, "dataFile %s is an compr_inv_table binary file", dataFile.c_str());
        exit(1);
    }
}


std::string convertTableToBinary(const std::string &dataFile, GPUGenie::GPUGenie_Config &config)
{
    std::string invSuffix(".inv");
    std::string cinvSuffix(".cinv");

    std::string invTableFileBase = dataFile.substr(0, dataFile.find_last_of('.'));
    std::string binaryInvTableFile;
    if (config.compression.empty())
        binaryInvTableFile = invTableFileBase + invSuffix;
    else
        binaryInvTableFile = invTableFileBase + std::string(".") + config.compression + cinvSuffix;

    Logger::log(Logger::INFO, "Converting table %s to %s (%s compression)...",
        dataFile.c_str(), binaryInvTableFile.c_str(), config.compression.empty() ? "no" : config.compression.c_str());

    ifstream invBinFileStream(binaryInvTableFile.c_str());
    bool invBinFileExists = invBinFileStream.good();

    if (invBinFileExists)
        Logger::log(Logger::INFO, "File %s already exists. Will ve overwritten!");
    invBinFileStream.close();


    Logger::log(Logger::INFO, "Reading data file %s ...", dataFile.c_str());
        
    read_file(dataFile.c_str(), &config.data, config.item_num, &config.index, config.row_num);
    assert(config.item_num > 0);
    assert(config.row_num > 0);
    Logger::log(Logger::DEBUG, "config.item_num: %d", config.item_num);
    Logger::log(Logger::DEBUG, "config.row_num: %d", config.row_num);


    Logger::log(Logger::INFO, "Preprocessing inverted table from %s ...", dataFile.c_str());
    inv_table * table = NULL;
    inv_compr_table * comprTable = NULL;
    preprocess_for_knn_binary(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != NULL);
    assert(table->build_status() == inv_table::builded);
    assert(table->get_total_num_of_table() == 1); // check how many tables we have

    if (!config.compression.empty()){
        comprTable = dynamic_cast<inv_compr_table*>(table);
        assert((int)comprTable->getUncompressedPostingListMaxLength() <= config.posting_list_max_length);
        // check the compression was actually used in the table
        assert(config.compression == comprTable->getCompression());
    }

    if (!inv_table::write(binaryInvTableFile.c_str(), table)) {
        Logger::log(Logger::ALERT, "Error writing inverted table to binary file %s!", binaryInvTableFile.c_str());
        return std::string();
    }

    Logger::log(Logger::INFO, "Sucessfully written inverted table to binary file %s.", binaryInvTableFile.c_str());
    return binaryInvTableFile;
}

void runGENIE(const std::string &binaryInvTableFile, const std::string &queryFile, GPUGenie::GPUGenie_Config &config,
        std::vector<int> &refResultIdxs, std::vector<int> &refResultCounts)
{
    Logger::log(Logger::INFO, "Opening binary inv_table from %s ...", binaryInvTableFile.c_str());

    inv_table *table;
    if (config.compression.empty()){
        inv_table::read(binaryInvTableFile.c_str(), table);
    }
    else {
        inv_compr_table *comprTable;
        inv_compr_table::read(binaryInvTableFile.c_str(), comprTable);
        table = comprTable;
    }

    Logger::log(Logger::INFO, "Loading queries from %s ...", queryFile.c_str());
    read_file(*config.query_points, queryFile.c_str(), config.num_of_queries);

    Logger::log(Logger::INFO, "Loading queries into table...");
    std::vector<query> refQueries;
    load_query(*table, refQueries, config);

    Logger::log(Logger::INFO, "Running KNN on GPU...");
    std::cout << "KNN_SEARCH_CPU"
              << ", file: " << binaryInvTableFile << " (" << config.row_num << " rows)" 
              << ", queryFile: " << queryFile << " (" << config.num_of_queries << " queries)"
              << ", topk: " << config.num_of_topk
              << ", compression: " << config.compression
              << ", ";

    knn_search(*table, refQueries, refResultIdxs, refResultCounts, config);
    // Top k results from GENIE don't have to be sorted. In order to compare with CPU implementation, we have to
    // sort the results manually from individual queries => sort subsequence relevant to each query independently
    sortGenieResults(config, refResultIdxs, refResultCounts);

    Logger::log(Logger::DEBUG, "Results from GENIE:");
    Logger::logResults(Logger::DEBUG, refQueries, refResultIdxs, refResultCounts);
}

int main(int argc, char* argv[])
{
    Logger::log(Logger::INFO, "Available compressions in GENIE (GPUGenie_Config::COMPRESSION_NAMES):");
    for (std::string &compr : GPUGenie_Config::COMPRESSION_NAMES)
        Logger::log(Logger::INFO, "  %s", compr.c_str());

    string dataFile = DEFAULT_TEST_DATASET;
    string queryFile = DEFAULT_QUERY_DATASET;
    
    vector<vector<int>> queryPoints;

    int dimensions = DEFAULT_DIMENSIONS;
    int numberOfQueries = DEFAULT_NUM_QUERIES;

    checkDataFileIsNotBinary(dataFile);

    GPUGenie_Config config;

    config.dim = dimensions;
    config.count_threshold = 14;
    config.num_of_topk = 10;
    config.hashtable_size = 100*config.num_of_topk*1.5;
    config.query_radius = 0;
    config.use_device = 2;
    config.use_adaptive_range = false;
    config.selectivity = 0.0f;

    config.query_points = &queryPoints;
    config.data_points = NULL;

    config.use_load_balance = true;
    config.posting_list_max_length = 1024;
    config.multiplier = 1.0f;
    config.use_multirange = false;

    config.data_type = 1;
    config.search_type = 0;
    config.max_data_size = 0;

    config.num_of_queries = numberOfQueries;

    std::string binaryInvTableFile = convertTableToBinary(dataFile, config);
    std::map<std::string, std::string> binaryComprInvTableFilesMap;
    for (std::string &compr : GPUGenie_Config::COMPRESSION_NAMES){
        config.compression = compr;
        binaryComprInvTableFilesMap[config.compression] = convertTableToBinary(dataFile, config);
    }


    init_genie(config);

    Logger::log(Logger::INFO, "Running GENIE with uncompressed table...");
    config.compression = std::string();    
    std::vector<int> refResultIdxs;
    std::vector<int> refResultCounts;
    runGENIE(binaryInvTableFile, queryFile, config, refResultIdxs, refResultCounts);


    Logger::log(Logger::INFO, "Running GENIE with compressed table...");
    config.compression = "copy4";
    std::string binaryInvComprTableFile = binaryComprInvTableFilesMap[config.compression];
    std::vector<int> resultIdxs;
    std::vector<int> resultCounts;
    runGENIE(binaryInvComprTableFile, queryFile, config, resultIdxs, resultCounts);


    Logger::log(Logger::INFO, "Comparing reference and compressed results...");
    // Compare the first docId from the GPU and CPU results -- note since we use points from the data file
    // as queries, One of the resutls is a full-dim count match (self match), which is what we compare here.
    // Note that for large datasets, the self match may not be included if config.num_of_topk is not high enough,
    // which is due to all the config.num_of_topk having count equal to config.dims (match in all dimensions),
    // thereby this test may fail for large datasets.
    assert(refResultIdxs[0 * config.num_of_topk] == resultIdxs[0 * config.num_of_topk]
        && refResultCounts[0 * config.num_of_topk] == resultCounts[0 * config.num_of_topk]);
    assert(refResultIdxs[1 * config.num_of_topk] == resultIdxs[1 * config.num_of_topk]
        && refResultCounts[1 * config.num_of_topk] == resultCounts[1 * config.num_of_topk]);
    assert(refResultIdxs[2 * config.num_of_topk] == resultIdxs[2 * config.num_of_topk]
        && refResultCounts[2 * config.num_of_topk] == resultCounts[2 * config.num_of_topk]);

    return 0;
}

