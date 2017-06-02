/**
 * \brief Performance measurement toolkit for compression of inverted lists in GENIE
 *
 */

#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <random>
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
#include <GPUGenie/DeviceSerialCodec.h>
#include <GPUGenie/DeviceBitPackingCodec.h>
#include <GPUGenie/DeviceVarintCodec.h>
#include <GPUGenie/scan.h> 

using namespace GPUGenie;
namespace po = boost::program_options;

 const int MAX_UNCOMPRESSED_LENGTH = 1024;


/**
 *  Generate data, where each number follows a geometrical distribution. Note that the array is not sorted and is
 *  likely to have many repeated values.
 *
 *  \param geom_distr_coeff coefficient of the geometrical distribution (e.g. 0.5)
 *  \param geom_distr_multiplier multiplicative base for numbers generated from the geometrical distribution
 *
 *  \return shared_ptr to the generated array
 */
std::shared_ptr<uint> generateGeometricInput(size_t length, double geom_distr_coeff, double geom_distr_multiplier, int seed)
{
    std::shared_ptr<uint> sp_h_Input(new uint[length], std::default_delete<uint[]>());

    std::default_random_engine gen(seed);
    std::geometric_distribution<int> gdist(geom_distr_coeff);

    for (int i = 0; i < (int)length; ++i)
        sp_h_Input.get()[i] = gdist(gen) * geom_distr_multiplier;

    return sp_h_Input;
}

/**
 *  Generate data, where the (positive) difference between two subsequent numbers follows a geometrical distribution.
 *  The distribution is already sorted.
 *
 *  \param geom_distr_coeff coefficient of the geometrical distribution (e.g. 0.5)
 *  \param geom_distr_multiplier multiplicative base for the deltas generated from the geometrical distribution
 *
 *  \return shared_ptr to the generated array
 */
std::shared_ptr<uint> generateGeometricDeltaInput(size_t length, double geom_distr_coeff, double geom_distr_multiplier, int seed)
{
    std::shared_ptr<uint> sp_h_Input(new uint[length], std::default_delete<uint[]>());

    std::default_random_engine gen(seed);
    std::geometric_distribution<int> gdist(geom_distr_coeff);

    int number = 0, delta;
    for (int i = 0; i < (int)length; ++i)
    {
        delta = (gdist(gen) * (geom_distr_multiplier)) + 1;
        if (number + delta < number)
        {
            Logger::log(Logger::ALERT,"Int overflow during generateRandomDeltaInput!");
            exit(4);
        }
        number += delta;
        sp_h_Input.get()[i] = number;
    }
    return sp_h_Input;
}


/**
 *  Generate data, where each value is generated UAR. The array is sorted and each value is unique.
 *
 *  \param minValue min value of the uniform distribution (inclusive)
 *  \param maxValue max value of the uniform distribution (inclusive)
 *
 *  \return shared_ptr to the generated array
 */
std::shared_ptr<uint> generateUniformInput(size_t length, int minValue, int maxValue, int seed)
{
    std::shared_ptr<uint> sp_h_Input(new uint[length], std::default_delete<uint[]>());

    std::default_random_engine gen(seed);
    std::uniform_int_distribution<> udist(minValue, maxValue);

    for (int i = 0; i < (int)length; ++i)
        sp_h_Input.get()[i] = udist(gen);

    std::sort(sp_h_Input.get(), sp_h_Input.get()+length);

    uint prev = 0;
    for (int i = 1; i < (int)length; ++i)
    {
        uint curr = sp_h_Input.get()[i];
        if (curr <= prev)
            curr = prev+1;
        prev = curr;
    }

    return sp_h_Input;
}



void printData(std::shared_ptr<uint> sp_h_Input, size_t length)
{
    std::copy(sp_h_Input.get(), sp_h_Input.get()+length, std::ostream_iterator<int>(std::cout,", "));
    std::cout << std::endl;
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

    if (comprLength > GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH) {
        // Codecs cannot decompress across mutliple blocks, where each block can only decompress lists of length at most
        // GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH
        Logger::log(Logger::DEBUG, "Codec: %s, Array size: %d, Compressed size: %d, Ratio: %.3f bpi, Compressed list too long",
            codec.name().c_str(), arrayLength, comprLength, 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength));
        ofs << codec.name() << ","
            << arrayLength << ","
            << comprLength << ","
            << std::fixed << std::setprecision(3) << 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength) << ","
            << std::fixed << std::setprecision(3) << 0 << ","
            << std::fixed << std::setprecision(3) << 0 << std::endl;
        return;
    }

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

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);
    
    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);
    
    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);
    
    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);
    
    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);
    
    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength, ofs);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>>
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

void openResultsFile(std::ofstream &ofs, const std::string &destDir, const std::string &measurement, int numOfRuns,
        double geom_distr_coeff, double geom_distr_multiplier, int srand)
{
    std::string sep("_");
    std::string fname(
            measurement+sep
            +"gd"+sep // geometric distribution
            +"p"+std::to_string(geom_distr_coeff)+sep
            +"m"+std::to_string(geom_distr_multiplier)+sep
            +"r"+std::to_string(numOfRuns)+sep
            +std::to_string(srand)+".csv");
    openResultsFile(ofs, destDir, fname);
}

void openResultsFile(std::ofstream &ofs, const std::string &destDir, const std::string &measurement, int numOfRuns,
        int minValue, int maxValue, int srand)
{
    std::string sep("_");
    std::string fname(
            measurement+sep
            +"ud"+sep // geometric distribution
            +"v"+std::to_string(minValue)+"-"+std::to_string(maxValue)+sep
            +"r"+std::to_string(numOfRuns)+sep
            +std::to_string(srand)+".csv");
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

std::string getBinaryFileName(const std::string &dataFile, COMPRESSION_TYPE compression)
{
    std::string invSuffix(".inv");
    std::string cinvSuffix(".cinv");

    std::string invTableFileBase = dataFile.substr(0, dataFile.find_last_of('.'));
    std::string binaryInvTableFile;
    if (!compression)
        binaryInvTableFile = invTableFileBase + invSuffix;
    else
        binaryInvTableFile = invTableFileBase + std::string(".") +
                DeviceCodecFactory::getCompressionName(compression) + cinvSuffix;
    return binaryInvTableFile;
}

void verifyTableCompression(GPUGenie::inv_compr_table *comprTable)
{
    std::vector<int> &inv = *(comprTable->uncompressedInv());
    std::vector<int> &invPos = *(comprTable->uncompressedInvPos());
    std::vector<uint32_t> &compressedInv = *(comprTable->compressedInv());
    std::vector<int> &compressedInvPos = *(comprTable->compressedInvPos());
    size_t maxUncomprLength = comprTable->getUncompressedPostingListMaxLength();


    std::shared_ptr<GPUGenie::DeviceIntegerCODEC> codec = DeviceCodecFactory::getCodec(comprTable->getCompression());

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
        (void)expectedUncomrLength;
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
        dataFile.c_str(), binaryInvTableFile.c_str(),
        !config.compression ? "no" : DeviceCodecFactory::getCompressionName(config.compression).c_str());

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

    if (config.compression){
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

    Logger::log(Logger::DEBUG, "Deallocating inverted table...");
    delete[] table;

    Logger::log(Logger::INFO, "Sucessfully written inverted table to binary file %s.", binaryInvTableFile.c_str());
    return binaryInvTableFile;
}

void runSingleGENIE(const std::string &binaryInvTableFile, const std::string &queryFile, GPUGenie::GPUGenie_Config &config,
        std::vector<int> &refResultIdxs, std::vector<int> &refResultCounts)
{
    Logger::log(Logger::INFO, "Opening binary inv_table from %s ...", binaryInvTableFile.c_str());

    GPUGenie::inv_table *table;
    if (!config.compression){
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
              << ", compression: " << (!config.compression ? "no" : DeviceCodecFactory::getCompressionName(config.compression).c_str())
              << std::endl;

    GPUGenie::knn_search(*table, refQueries, refResultIdxs, refResultCounts, config);
    // Top k results from GENIE don't have to be sorted. In order to compare with CPU implementation, we have to
    // sort the results manually from individual queries => sort subsequence relevant to each query independently
    sortGenieResults(config, refResultIdxs, refResultCounts);

    if (!config.compression){
        GPUGenie::PerfLogger::get().ofs() << std::fixed << std::setprecision(3) << 0.0 << std::endl; // "comprRatio"
    }
    else {
        GPUGenie::inv_compr_table *comprTable = dynamic_cast<GPUGenie::inv_compr_table*>(table);
        GPUGenie::PerfLogger::get().ofs() << std::fixed << std::setprecision(3)
                << comprTable->getCompressionRatio() << std::endl; // "comprRatio"
    }

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
    config.use_device = 1;
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


    for (COMPRESSION_TYPE compr : DeviceCodecFactory::allCompressionTypes)
    {
        if (codec != "all" && codec != DeviceCodecFactory::getCompressionName(compr))
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
        << "codec" << ","
        << "overallTime" << ","
        << "queryCompilationTime" << ","
        << "preprocessingTime" << ","
        << "queryTransferTime" << ","
        << "dataTransferTime" << ","
        << "constantTransferTime" << ","
        << "allocationTime" << ","
        << "fillingTime" << ","
        << "matchingTime" << ","
        << "convertTime" << ","
        << "comprRatio" << std::endl;

    GPUGenie::init_genie(config);

    if (codec == "all" || codec == "no")
        Logger::log(Logger::INFO, "Running GENIE with uncompressed table...");
    else
        Logger::log(Logger::INFO,
            "Running GENIE with uncompressed table (to establish reference solution for codec %s", codec.c_str());
    config.compression = NO_COMPRESSION;    
    std::vector<int> refResultIdxs;
    std::vector<int> refResultCounts;
    runSingleGENIE(getBinaryFileName(dataFile, config.compression), queryFile, config, refResultIdxs, refResultCounts);

    for (COMPRESSION_TYPE compr : DeviceCodecFactory::allCompressionTypes)
    {
        if (codec != "all" && codec != DeviceCodecFactory::getCompressionName(compr))
            continue; // Skip this codec

        Logger::log(Logger::INFO, "Running GENIE with compressed (%s) table...",
                DeviceCodecFactory::getCompressionName(compr).c_str());

        config.compression = compr;
        std::vector<int> resultIdxs;
        std::vector<int> resultCounts;
        runSingleGENIE(getBinaryFileName(dataFile, config.compression), queryFile, config, resultIdxs, resultCounts);

        Logger::log(Logger::INFO, "Comparing reference and compressed results...");
        // Compare the first docId from the GPU and CPU results -- note since we use points from the data file
        // as queries, One of the resutls is a full-dim count match (self match), which is what we compare here.
        // Note that for large datasets, the self match may not be included if config.num_of_topk is not high enough,
        // which is due to all the config.num_of_topk having count equal to config.dims (match in all dimensions),
        // thereby if the last k-th result count is the same as the first, we do not compare the result idx, otherwise
        // the test would likely fail. GENIE does not guarantee that the result set will consist of lowest idx among
        // results of the same count.
        assert(refResultCounts[0 * config.num_of_topk] == resultCounts[0 * config.num_of_topk]);
        assert(refResultCounts[1 * config.num_of_topk] == resultCounts[1 * config.num_of_topk]);
        assert(refResultCounts[2 * config.num_of_topk] == resultCounts[2 * config.num_of_topk]);
        // Only compare idxs in case the counts of the first result and k-th result are different
        assert(refResultCounts[0 * config.num_of_topk] == refResultCounts[1 * config.num_of_topk -1] ||
            refResultIdxs[0 * config.num_of_topk] == resultIdxs[0 * config.num_of_topk]);
        assert(refResultCounts[1 * config.num_of_topk] == refResultCounts[2 * config.num_of_topk -1] ||
            refResultIdxs[1 * config.num_of_topk] == resultIdxs[1 * config.num_of_topk]);
        assert(refResultCounts[2 * config.num_of_topk] == refResultCounts[3 * config.num_of_topk -1] ||
            refResultIdxs[2 * config.num_of_topk] == resultIdxs[2 * config.num_of_topk]);
    }

}

int main(int argc, char **argv)
{    
    Logger::log(Logger::INFO,"Running %s", argv[0]);

    double geom_distr_coeff, geom_distr_multiplier;
    int srand, numRuns, uniform_distr_min, uniform_distr_max;
    std::string dest, data, codec;
    std::vector<std::string> operations;

    po::options_description desc("Compression performance measurements");
    desc.add_options()
        ("help",
            "produce help message")
        ("runs", po::value<int>(&numRuns)->default_value(1),
            "number of runs of each test (to average the results)")
        ("srand", po::value<int>(&srand)->default_value(666),
            "seed for input data generator")
        ("uniform_distr_min", po::value<int>(&uniform_distr_min)->default_value(1),
            "minimum value generated by uniform distribution")
        ("uniform_distr_max", po::value<int>(&uniform_distr_max)->default_value(1<<16),
            "maximum value generated by uniform distribution")
        ("geom_distr_coeff", po::value<double>(&geom_distr_coeff)->default_value(0.005),
            "coefficient for geometric distribution (known as 'p')")
        ("geom_distr_multiplier", po::value<double>(&geom_distr_multiplier)->default_value(1.0),
            "multiplicative coefficient for numbers generated from the geometric_distribution")
        ("operation", po::value< std::vector<std::string>>(&operations),
            "operation to run, one from: {scan, codecs, separate, integrated, convert}")
        ("data", po::value<std::string>(&data),
            "* when operation = \"scan\" or \"codecs\", data determines distribution to use, one from {geometric, uniform}\n"
            "* when operation = \"separate\", \"integrated\" or \"convert\", data sets a path to csv file, currently"
            " supported configurations only for filenames: {*adult.csv, *ocr.csv, *sift_20.csv, *sift_4.5m.csv, *tweets_20.csv, *tweets.csv}")
        ("codec", po::value<std::string>(&codec)->default_value(std::string("all")),
            "codec to use (works with \"integrated\" and \"convert\" operations), one from: {all, no, copy, d1, bp32, varint, bp32-copy, bp32-varint}")
        ("dest", po::value<std::string>(&dest)->default_value(std::string("../results")),
            "destination directory");

    po::positional_options_description pdesc;
    pdesc.add("operation", 1);
    pdesc.add("data", 1);
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

    if (!vm.count("data"))
    {
        std::cerr << "Data not specified!" << std::endl;
        return 1;
    }

    if (numRuns != 1)
    {
        std::cerr << "Invalid numbers of runs" << std::endl;
        return 1;
    }

    std::ofstream ofs;
    for (auto it = operations.begin(); it != operations.end(); it++)
    {
        if (*it == std::string("scan"))
        {
            std::shared_ptr<uint> h_Input;
            if (data == std::string("geometric"))
            {
                h_Input = (generateGeometricDeltaInput(SCAN_MAX_LARGE_ARRAY_SIZE, geom_distr_coeff, geom_distr_multiplier, srand));
            }
            else if (data == std::string("uniform"))
            {
                if (uniform_distr_max - uniform_distr_min <= (int)SCAN_MAX_LARGE_ARRAY_SIZE)
                {
                    std::cerr << "Uniform distribution error, min and max range too narrow!" << std::endl;
                    return 1;
                }
                h_Input = (generateUniformInput(SCAN_MAX_LARGE_ARRAY_SIZE, uniform_distr_min, uniform_distr_max, srand));
            }
            else 
            {
                std::cerr << "Unknown distribution " << data << std::endl;
                return 1;
            }
            // printData(h_Input, 2048);
            openResultsFile(ofs, dest, *it, numRuns, geom_distr_coeff, geom_distr_multiplier, srand);
            measureScan(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("codecs"))
        {
            std::shared_ptr<uint> h_Input;
            if (data == std::string("geometric"))
            {
                h_Input = generateGeometricDeltaInput(MAX_UNCOMPRESSED_LENGTH, geom_distr_coeff, geom_distr_multiplier, srand);
            }
            else if (data == std::string("uniform"))
            {
                if (uniform_distr_max - uniform_distr_min <= MAX_UNCOMPRESSED_LENGTH)
                {
                    std::cerr << "Uniform distribution error, min and max values too narrow!" << std::endl;
                    return 1;
                }
                h_Input = generateUniformInput(MAX_UNCOMPRESSED_LENGTH, uniform_distr_min, uniform_distr_max, srand);
            }
            else 
            {
                std::cerr << "Unknown distribution " << data << std::endl;
                return 1;
            }
            // printData(h_Input, MAX_UNCOMPRESSED_LENGTH);
            openResultsFile(ofs, dest, *it, numRuns, geom_distr_coeff, geom_distr_multiplier, srand);
            measureCodecs(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("separate"))
        {
            std::cerr << "separate kernel operation not yet implemented" << std::endl;
            return 1;
        }
        else if (*it == std::string("integrated"))
        {
            if (!vm.count("data"))
            {
                std::cerr << "Measurement \"intergrated\" requires a data argument!" << std::endl;
                return 1;
            }
            openResultsFile(ofs, dest, *it, data);
            runGENIE(data, codec, ofs);
            ofs.close();
        }
        else if (*it == std::string("convert"))
        {
            if (!vm.count("data"))
            {
                std::cerr << "Operation \"convert\" requires a data argument!" << std::endl;
                return 1;
            }
            convertTableToBinaryFormats(data, codec);
        }
        else
        {
            std::cerr << "Unknown operation: " << *it << std::endl;
            return 1;
        }
    }
    return 0;
}
