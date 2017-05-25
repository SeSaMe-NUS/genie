/**
 * \brief Performance measurement toolkit for compression of inverted lists in GENIE
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>

#include <boost/program_options.hpp>

#include <GPUGenie/genie_errors.h>
#include <GPUGenie/interface.h>
#include <GPUGenie/Timing.h>
#include <GPUGenie/PerfLogger.hpp>
#include <GPUGenie/Logger.h>
#include <GPUGenie/DeviceCompositeCodec.h>
#include <GPUGenie/DeviceBitPackingCodec.h>
#include <GPUGenie/DeviceVarintCodec.h>
#include <GPUGenie/scan.h> 

using namespace GPUGenie;
namespace po = boost::program_options;

 const int MAX_UNCOMPRESSED_LENGTH = 1024;

std::shared_ptr<uint> generateRandomInput(size_t length, double geom_distr_coeff, int seed)
{
    std::shared_ptr<uint> sp_h_Input(new uint[length], std::default_delete<uint[]>());
    uint *h_Input = sp_h_Input.get();

    srand(seed);

    assert(sizeof(long int) >= 8); // otherwise there may be an overflow in these generated numbers
    for (uint i = 0; i < length; i++)
    {
        switch (rand() % 5)
        {
            case 0: // generate 1-7 bit number
                h_Input[i] = (long int)rand() % (1 << 7);
                break;
            case 1: // generate 8-14 bit number
                h_Input[i] = ((long int)rand() + (1 << 7)) % (1 << 15);
                break;
            case 2: // generate 15-21 bit number
                h_Input[i] = ((long int)rand() + (1 << 14)) % (1 << 22);
                break;
            case 3: // generate 22-28 bit number
                h_Input[i] = ((long int)rand() + (1 << 21)) % (1 << 28);
                break;
            case 4: // generate 29-32 bit number
                h_Input[i] = ((long int)rand() + (1 << 28));
                break;
        }
        
    }
    return sp_h_Input;
}

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

void runSingleScan(uint *h_Input, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_Input, uint *d_Output,
    size_t arrayLength, std::ofstream &ofs)
{
    cudaCheckErrors(cudaMemset(d_Output, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));

    memset(h_OutputCPU, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    
    uint64_t scanStart = getTime();
    if (arrayLength <= SCAN_MAX_SHORT_ARRAY_SIZE)
        scanExclusiveShort(d_Output, d_Input, arrayLength);
    else
        scanExclusiveLarge(d_Output, d_Input, arrayLength);
    cudaCheckErrors(cudaDeviceSynchronize());
    uint64_t scanEnd = getTime();

    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint),
        cudaMemcpyDeviceToHost));

    scanExclusiveHost(h_OutputCPU, h_Input, arrayLength);

    double scanTime = getInterval(scanStart, scanEnd);
    Logger::log(Logger::DEBUG, "Scan, Array size: %d, Time: %.3f ms, Throughput: %.3f elements per millisecond"
        , arrayLength, scanTime, (double)arrayLength/scanTime);
    ofs << arrayLength << ","
        << std::fixed << std::setprecision(3) << scanTime << ","
        << std::fixed << std::setprecision(3) <<(double)arrayLength/scanTime << std::endl;
}

void measureScan(std::shared_ptr<uint> sp_h_Input, std::ofstream &ofs)
{
    uint *d_Input, *d_Output;
    uint *h_Input = sp_h_Input.get(), *h_OutputCPU, *h_OutputGPU;

    Logger::log(Logger::DEBUG,"Allocating and initializing CPU & CUDA arrays...\n");

    h_OutputCPU  = (uint *)malloc(SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    h_OutputGPU  = (uint *)malloc(SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    cudaCheckErrors(cudaMalloc((void **)&d_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));

    // Copy input to GPU
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));

    Logger::log(Logger::DEBUG,"Measuring scan...\n\n");

    ofs << "array_size,time,throughput" << std::endl;
    initScan();
    for (int length = 4; length <= (int)SCAN_MAX_SHORT_ARRAY_SIZE; length += 4){
        runSingleScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, length, ofs);
    }
    closeScan();

    free(h_OutputCPU);
    free(h_OutputGPU);
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
}



template <class CODEC>
void runSingleCodec(uint *h_Input, uint *h_InputCompr, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_InputCompr,
        uint *d_Output, size_t arrayLength, size_t *d_decomprLength, std::ostream &ofs)
{
    CODEC codec;
    Logger::log(Logger::DEBUG,"\n\nTesting codec...\n\n",codec.name().c_str());

    size_t comprLength = MAX_UNCOMPRESSED_LENGTH;
    memset(h_InputCompr, 0, MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    codec.encodeArray(h_Input, arrayLength, h_InputCompr, comprLength);

    // Copy compressed array to GPU
    cudaCheckErrors(cudaMemcpy(d_InputCompr, h_InputCompr, MAX_UNCOMPRESSED_LENGTH * sizeof(uint), cudaMemcpyHostToDevice));
    // Clear working memory on both GPU and CPU
    cudaCheckErrors(cudaMemset(d_Output, 0, MAX_UNCOMPRESSED_LENGTH * sizeof(uint)));
    cudaCheckErrors(cudaMemset(d_decomprLength, 0, sizeof(size_t)));
    memset(h_OutputCPU, 0, MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    
    int threadsPerBlock = (codec.decodeArrayParallel_lengthPerBlock() / codec.decodeArrayParallel_threadLoad());
    int blocks = (arrayLength + threadsPerBlock * codec.decodeArrayParallel_threadLoad() - 1) /
            (threadsPerBlock * codec.decodeArrayParallel_threadLoad());
    assert(blocks <= codec.decodeArrayParallel_maxBlocks());

    // run decompression on GPU
    uint64_t decomprStart = getTime();
    g_decodeArrayParallel<CODEC><<<blocks,threadsPerBlock>>>(d_InputCompr, comprLength, d_Output, SCAN_MAX_SHORT_ARRAY_SIZE, d_decomprLength);
    cudaCheckErrors(cudaDeviceSynchronize());
    uint64_t decomprEnd = getTime();

    // copy decompression results from GPU
    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, MAX_UNCOMPRESSED_LENGTH * sizeof(uint), cudaMemcpyDeviceToHost));
    size_t decomprLengthGPU;
    cudaCheckErrors(cudaMemcpy(&decomprLengthGPU, d_decomprLength, sizeof(size_t), cudaMemcpyDeviceToHost));

    // run decompression on CPU
    size_t decomprLengthCPU = MAX_UNCOMPRESSED_LENGTH;
    codec.decodeArray(h_InputCompr, comprLength, h_OutputCPU, decomprLengthCPU);

    // Check if lenth from CPU and GPU decompression matches the original length
    assert(decomprLengthCPU == arrayLength);
    assert(decomprLengthGPU == arrayLength);

    double decomprTime = getInterval(decomprStart, decomprEnd);
    Logger::log(Logger::DEBUG, "Codec: %s, Array size: %d, Compressed size: %d, Ratio: %.3f bpi, Time (decompr): %.3f ms, Throughput: %.3f of elements per millisecond",
            codec.name().c_str(), arrayLength, comprLength, 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength),
            decomprTime, (double)arrayLength/decomprTime);
    ofs << codec.name() << ","
        << arrayLength << ","
        << comprLength << ","
        << std::fixed << std::setprecision(3) << 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength) << ","
        << std::fixed << std::setprecision(3) << decomprTime << ","
        << std::fixed << std::setprecision(3) << (double)arrayLength/decomprTime << std::endl;
}


void measureCodecs(std::shared_ptr<uint> sp_h_Input, std::ofstream &ofs)
{
    uint *d_Input, *d_Output;
    size_t *d_decomprLength;
    uint *h_Input = sp_h_Input.get(), *h_InputCompr, *h_OutputCPU, *h_OutputGPU;

    Logger::log(Logger::DEBUG,"Allocating and initializing CPU & CUDA arrays...\n");
    
    h_InputCompr = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    h_OutputCPU  = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    h_OutputGPU  = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    cudaCheckErrors(cudaMalloc((void **)&d_Input, MAX_UNCOMPRESSED_LENGTH * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, MAX_UNCOMPRESSED_LENGTH * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_decomprLength, sizeof(size_t)));

    // Copy input to GPU
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, MAX_UNCOMPRESSED_LENGTH * sizeof(uint), cudaMemcpyHostToDevice));

    ofs << "codec,array_size,compr_size,ratio,time,throughput" << std::endl;

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceCopyCodec>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceDeltaCodec>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceBitPackingCodec>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceVarintCodec>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);


    free(h_OutputCPU);
    free(h_OutputGPU);
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
    cudaCheckErrors(cudaFree(d_decomprLength));
}

void openResultsFile(std::ofstream &ofs, const std::string &destDir, const std::string &fileName)
{
    struct stat sb;
    if (stat(destDir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    {
        Logger::log(Logger::ALERT,"Destination directory %s is not a directory!\n", destDir.c_str());
        exit(2);
    }
    std::string dirsep("/");
    std::string fullPath(destDir+dirsep+fileName);
    Logger::log(Logger::INFO,"Output file: %s", fileName.c_str());

    ofs.open(fullPath.c_str(), std::ios_base::out | std::ios_base::trunc);
    GPUGenie::PerfLogger::get().setOutputFileStream(ofs);
    assert(ofs.is_open());
}

void openResultsFile(std::ofstream &ofs, const std::string &destDir, const std::string &measurement,
        int srand, double geom_distr_coeff)
{
    std::string sep("_");
    std::string fname(measurement+sep+std::to_string(srand)+sep+std::to_string(geom_distr_coeff)+".csv");
    openResultsFile(ofs, destDir, fname);
}

void openResultsFile(std::ofstream &ofs, const std::string &destDir, const std::string &measurement,
        const std::string &dataset)
{
    std::string sep("_"), dirsep("/"), datasetFilename = dataset;

    // Get base filename
    size_t lastDirSep = dataset.find_last_of(dirsep);
    if (lastDirSep != std::string::npos)
        datasetFilename = dataset.substr(lastDirSep+1);

    // Remove file extension if dataset includes that
    size_t lastDot = datasetFilename.find_last_of(".");
    if (lastDot != std::string::npos)
        datasetFilename = datasetFilename.substr(0, lastDot);

    if (datasetFilename.find_last_of(".") != std::string::npos) // Still contains a '.'
        Logger::log(Logger::ALERT,"Output file may be incorrectly generated!\n", destDir.c_str());

    std::string fname(measurement+sep+datasetFilename+".csv");
    openResultsFile(ofs, destDir, fname);
}

std::string getBinaryFileName(const std::string &dataFile, const std::string &compression)
{
    std::string invSuffix(".inv");
    std::string cinvSuffix(".cinv");

    std::string invTableFileBase = dataFile.substr(0, dataFile.find_last_of('.'));
    std::string binaryInvTableFile;
    if (compression.empty())
        binaryInvTableFile = invTableFileBase + invSuffix;
    else
        binaryInvTableFile = invTableFileBase + std::string(".") + compression + cinvSuffix;
    return binaryInvTableFile;
}

void verifyTableCompression(GPUGenie::inv_compr_table *comprTable)
{
    std::vector<int> &inv = *(comprTable->uncompressedInv());
    std::vector<int> &invPos = *(comprTable->uncompressedInvPos());
    std::vector<uint32_t> &compressedInv = *(comprTable->compressedInv());
    std::vector<int> &compressedInvPos = *(comprTable->compressedInvPos());
    size_t maxUncomprLength = comprTable->getUncompressedPostingListMaxLength();


    std::shared_ptr<GPUGenie::DeviceIntegerCODEC> codec = GPUGenie::inv_compr_table::getCodec(comprTable->getCompression());

    // A vector for decompressing inv lists
    std::vector<uint32_t> uncompressedInv(maxUncomprLength,0);
    uint32_t *out = uncompressedInv.data();
    for (int pos = 0; pos < (int)compressedInvPos.size()-1; pos++)
    {
        int comprInvStart = compressedInvPos[pos];
        int comprInvEnd = compressedInvPos[pos+1];
        assert(comprInvEnd - comprInvStart > 0 && comprInvEnd - comprInvStart <= (int)maxUncomprLength);

        uint32_t * in = compressedInv.data() + comprInvStart;
        size_t uncomprLength = maxUncomprLength;
        codec->decodeArray(in, comprInvEnd - comprInvStart, out, uncomprLength);

        // Check if the compressed length (uncomprLength) from encodeArray(...) does not exceed the max_length constraint
        // of the compressed list
        assert(uncomprLength > 0 && uncomprLength <= maxUncomprLength);

        int uncomprInvStart = invPos[pos];
        int uncomprInvEnd = invPos[pos+1];
        int expectedUncomrLength = uncomprInvEnd - uncomprInvStart;
        assert(expectedUncomrLength == (int)uncomprLength);

        for (int i = 0; i < (int)uncomprLength; i++){
            assert((int)uncompressedInv[i] == inv[uncomprInvStart+i]);
        }

    }
}


std::string convertTableToBinary(const std::string &dataFile, GPUGenie::GPUGenie_Config &config)
{
    std::string binaryInvTableFile = getBinaryFileName(dataFile, config.compression);

    Logger::log(Logger::INFO, "Converting table %s to %s (%s compression)...",
        dataFile.c_str(), binaryInvTableFile.c_str(), config.compression.empty() ? "no" : config.compression.c_str());

    std::ifstream invBinFileStream(binaryInvTableFile.c_str());
    bool invBinFileExists = invBinFileStream.good();
    invBinFileStream.close();

    if (invBinFileExists)
        Logger::log(Logger::INFO, "File %s already exists. Will ve overwritten!");

    Logger::log(Logger::INFO, "Preprocessing inverted table from %s ...", dataFile.c_str());
    GPUGenie::inv_table * table = nullptr;
    GPUGenie::inv_compr_table * comprTable = nullptr;
    GPUGenie::preprocess_for_knn_csv(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != nullptr);
    assert(table->build_status() == inv_table::builded);
    assert(table->get_total_num_of_table() == 1); // check how many tables we have

    if (!config.compression.empty()){
        comprTable = dynamic_cast<GPUGenie::inv_compr_table*>(table);
        assert((int)comprTable->getUncompressedPostingListMaxLength() <= config.posting_list_max_length);
        // check the compression was actually used in the table
        assert(config.compression == comprTable->getCompression());
        // Check decompression on CPU
        Logger::log(Logger::INFO, "Verifying compression on CPU...", dataFile.c_str());
        verifyTableCompression(comprTable);
    }

    if (!inv_table::write(binaryInvTableFile.c_str(), table)) {
        Logger::log(Logger::ALERT, "Error writing inverted table to binary file %s!", binaryInvTableFile.c_str());
        return std::string();
    }

    Logger::log(Logger::INFO, "Sucessfully written inverted table to binary file %s.", binaryInvTableFile.c_str());
    return binaryInvTableFile;
}

void runSingleGENIE(const std::string &binaryInvTableFile, const std::string &queryFile, GPUGenie::GPUGenie_Config &config,
        std::vector<int> &refResultIdxs, std::vector<int> &refResultCounts)
{
    Logger::log(Logger::INFO, "Opening binary inv_table from %s ...", binaryInvTableFile.c_str());

    GPUGenie::inv_table *table;
    if (config.compression.empty()){
        GPUGenie::inv_table::read(binaryInvTableFile.c_str(), table);
    }
    else {
        GPUGenie::inv_compr_table *comprTable;
        GPUGenie::inv_compr_table::read(binaryInvTableFile.c_str(), comprTable);
        table = comprTable;
    }

    Logger::log(Logger::INFO, "Loading queries from %s ...", queryFile.c_str());
    GPUGenie::read_file(*config.query_points, queryFile.c_str(), config.num_of_queries);

    Logger::log(Logger::INFO, "Loading queries into table...");
    std::vector<query> refQueries;
    GPUGenie::load_query(*table, refQueries, config);

    Logger::log(Logger::INFO, "Running KNN on GPU...");
    std::cout << "KNN_SEARCH_CPU"
              << ", file: " << binaryInvTableFile << " (" << config.row_num << " rows)" 
              << ", queryFile: " << queryFile << " (" << config.num_of_queries << " queries)"
              << ", topk: " << config.num_of_topk
              << ", compression: " << (config.compression.empty() ? "no" : config.compression.c_str()) << std::endl;

    GPUGenie::knn_search(*table, refQueries, refResultIdxs, refResultCounts, config);
    // Top k results from GENIE don't have to be sorted. In order to compare with CPU implementation, we have to
    // sort the results manually from individual queries => sort subsequence relevant to each query independently
    sortGenieResults(config, refResultIdxs, refResultCounts);

    Logger::log(Logger::DEBUG, "Results from GENIE:");
    Logger::logResults(Logger::DEBUG, refQueries, refResultIdxs, refResultCounts);

    Logger::log(Logger::DEBUG, "Deallocating inverted table...");
    delete[] table;
}


void fillConfig(const std::string &dataFile, GPUGenie::GPUGenie_Config &config)
{

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

    size_t lastDirSep = dataFile.find_last_of('/');
    std::string dataFileName = dataFile;
    if (lastDirSep != std::string::npos)
        dataFileName = dataFile.substr(lastDirSep+1);

    Logger::log(Logger::INFO, "Generating GENIE configuration for %s", dataFileName.c_str());

    if (dataFileName == std::string("sift_20.csv")){
        config.dim = 5;
        config.hashtable_size = 100*config.num_of_topk*1.5;
        config.search_type = 0;

    } else if (dataFileName == std::string("sift_4.5m.csv")){
        config.dim = 128;
        config.hashtable_size = 200*config.num_of_topk*1.5;
        config.search_type = 0;

    } else if (dataFileName == std::string("tweets_20.csv")){
        config.dim = 14;
        config.hashtable_size = 100*config.num_of_topk*1.5;
        config.search_type = 1;

    } else if (dataFileName == std::string("tweets.csv")){
        config.dim = 128;
        config.hashtable_size = 200*config.num_of_topk*1.5;
        config.search_type = 1;

    } else if (dataFileName == std::string("ocr.csv")){
        config.dim = 237;
        config.hashtable_size = 237*config.num_of_topk*1.5;
        config.search_type = 0;

    } else if (dataFileName == std::string("adult.csv")){
        config.dim = 14;
        config.hashtable_size = 14*config.num_of_topk*1.5;
        config.search_type = 0;

    } else {
        Logger::log(Logger::ALERT,"Unknown data file %s, cannot automatically generate GENIE configuration!\n",
            dataFileName.c_str());
        exit(3);
    }
    config.count_threshold = config.dim;
    config.query_radius = 0;
    config.use_adaptive_range = false;
    config.selectivity = 0.0f;
    config.use_load_balance = true;
    config.posting_list_max_length = 1024;
    config.use_multirange = false;
    config.data_type = 0;
    config.multiplier = 1.0f;
    config.use_device = 2;
    config.query_points = nullptr;
    config.data_points = nullptr;
}


void convertTableToBinaryFormats(const std::string &dataFile, const std::string &codec)
{
    GPUGenie::GPUGenie_Config config;
    fillConfig(dataFile, config);

    Logger::log(Logger::INFO, "Reading data file %s ...", dataFile.c_str());
    std::vector<std::vector<int>> data;
    config.data_points = &data;
    GPUGenie::read_file(data, dataFile.c_str(), -1);

    if (codec == "all" || codec == "no")
        std::string binaryInvTableFile = convertTableToBinary(dataFile, config);


    for (std::string &compr : GPUGenie::GPUGenie_Config::COMPRESSION_NAMES)
    {
        if (codec != "all" && codec != compr)
            continue; // Skip this codec
        config.compression = compr;
        convertTableToBinary(dataFile, config);
    }
}

void runGENIE(const std::string &dataFile, const std::string &codec, std::ostream &ofs)
{
    std::string queryFile = dataFile;
    vector<vector<int>> queryPoints;

    GPUGenie::GPUGenie_Config config;
    config.num_of_queries = 5;
    config.num_of_topk = 5;
    fillConfig(dataFile, config);
    config.query_points = &queryPoints;

    GPUGenie::PerfLogger::get().ofs()
        << "overallTime" << ","
        << "queryCompilationTime" << ","
        << "preprocessingTime" << ","
        << "queryTransferTime" << ","
        << "dataTransferTime" << ","
        << "constantTransferTime" << ","
        << "allocationTime" << ","
        << "fillingTime" << ","
        << "matchingTime" << ","
        << "convertTime" << std::endl;

    GPUGenie::init_genie(config);

    if (codec == "all" || codec == "no")
        Logger::log(Logger::INFO, "Running GENIE with uncompressed table...");
    else
        Logger::log(Logger::INFO,
            "Running GENIE with uncompressed table (to establish reference solution for codec %s", codec.c_str());
    config.compression = std::string();    
    std::vector<int> refResultIdxs;
    std::vector<int> refResultCounts;
    runSingleGENIE(getBinaryFileName(dataFile, config.compression), queryFile, config, refResultIdxs, refResultCounts);

    for (std::string &compr : GPUGenie::GPUGenie_Config::COMPRESSION_NAMES)
    {
        if (codec != "all" && codec != compr)
            continue; // Skip this codec

        Logger::log(Logger::INFO, "Running GENIE with compressed (%s) table...",compr.c_str());

        config.compression = compr;
        std::vector<int> resultIdxs;
        std::vector<int> resultCounts;
        runSingleGENIE(getBinaryFileName(dataFile, config.compression), queryFile, config, resultIdxs, resultCounts);

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
    }

}

int main(int argc, char **argv)
{    
    Logger::log(Logger::INFO,"Running %s", argv[0]);

    double geom_distr_coeff;
    int srand;
    std::string dest, datafile, codec;
    std::vector<std::string> operations;

    po::options_description desc("Compression performance measurements");
    desc.add_options()
        ("help",
            "produce help message")
        ("srand", po::value<int>(&srand)->default_value(666),
            "seed for input data generator")
        ("geom_distr_coeff", po::value<double>(&geom_distr_coeff)->default_value(0.9),
            "coefficient for geometric distribution")
        ("operation", po::value< std::vector<std::string>>(&operations),
            "operation to run, one from: {scan, codecs, separate, integrated, convert}")
        ("datafile", po::value<std::string>(&datafile),
            "path to datafile file, currently supported filenames: {*adult.csv, *ocr.csv, *sift_20.csv, *sift_4.5m.csv, *tweets_20.csv, *tweets.csv}")
        ("codec", po::value<std::string>(&codec)->default_value(std::string("all")),
            "codec to use (works with \"integrated\" and \"convert\" operations), one from: {all, no, copy, d1, bp32, varint, bp32-copy, bp32-varint}")
        ("dest", po::value<std::string>(&dest)->default_value(std::string("../results")),
            "destination directory");

    po::positional_options_description pdesc;
    pdesc.add("operation", 1);
    pdesc.add("datafile", 1);
    pdesc.add("codec", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv). options(desc).positional(pdesc).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    if (!vm.count("operation"))
    {
        std::cerr << "No operation to run" << std::endl;
        return 1;
    }

    if (vm.count("srand"))
    {
        std::cerr << "srand not yet implemented" << std::endl;
    }

    if (vm.count("geom_distr_coeff"))
    {
        std::cerr << "geom_distr_coeff not yet implemented" << std::endl;
    }

    std::ofstream ofs;
    for (auto it = operations.begin(); it != operations.end(); it++)
    {
        if (*it == std::string("scan")){
            std::shared_ptr<uint> h_Input = generateRandomInput(SCAN_MAX_LARGE_ARRAY_SIZE, geom_distr_coeff, srand);
            openResultsFile(ofs, dest, *it, srand, geom_distr_coeff);
            measureScan(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("codecs")){
            std::shared_ptr<uint> h_Input = generateRandomInput(SCAN_MAX_LARGE_ARRAY_SIZE, geom_distr_coeff, srand);
            openResultsFile(ofs, dest, *it, srand, geom_distr_coeff);
            measureCodecs(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("separate")){
            std::cerr << "separate kernel operation not yet implemented" << std::endl;
            return 1;
        }
        else if (*it == std::string("integrated")){
            if (!vm.count("datafile"))
            {
                std::cerr << "Measurement \"intergrated\" requires a datafile argument!" << std::endl;
                return 1;
            }
            openResultsFile(ofs, dest, *it, datafile);
            runGENIE(datafile, codec, ofs);
            ofs.close();
        }
        else if (*it == std::string("convert")){
            if (!vm.count("datafile"))
            {
                std::cerr << "Operation \"convert\" requires a datafile argument!" << std::endl;
                return 1;
            }
            convertTableToBinaryFormats(datafile, codec);
        }
        else {
            std::cerr << "Unknown operation: " << *it << std::endl;
            return 1;
        }
    }
    return 0;
}
