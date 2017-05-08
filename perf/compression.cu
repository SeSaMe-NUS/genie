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
#include <GPUGenie/Timing.h>
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
        runSingleCodec<DeviceJustCopyCodec>
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
        runSingleCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceJustCopyCodec>>
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

void openOutputFile(std::ofstream &ofs, const std::string &dest, const std::string &measurement,
        int srand, double geom_distr_coeff)
{
    struct stat sb;
    if (stat(dest.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    {
        Logger::log(Logger::ALERT,"--dest=%s is not a directory!\n", dest.c_str());
        exit(2);
    }
    std::string sep("_"), dirsep("/");
    std::string fname(dest+dirsep+measurement+sep+std::to_string(srand)+sep+std::to_string(geom_distr_coeff)+".csv");
    Logger::log(Logger::INFO,"Output file: %s \n\n", fname.c_str());

    ofs.open(fname.c_str(), std::ios_base::out | std::ios_base::trunc);
    assert(ofs.is_open());
}

int main(int argc, char **argv)
{    
    Logger::log(Logger::INFO,"Running %s \n\n", argv[0]);

    double geom_distr_coeff;
    int srand;
    std::string dest;
    std::vector<std::string> measurements;

    po::options_description desc("Compression performance measurements");
    desc.add_options()
        ("help",
            "produce help message")
        ("srand", po::value<int>(&srand)->default_value(666),
            "seed for input data generator")
        ("geom_distr_coeff", po::value<double>(&geom_distr_coeff)->default_value(0.9),
            "coefficient for geometric distribution")
        ("measurements", po::value< std::vector<std::string>>(&measurements),
            "space separated measurements from: scan, codecs, separate, integrated")
        ("dest", po::value<std::string>(&dest)->default_value(std::string("../results")),
            "destination directory");

    po::positional_options_description pdesc;
    pdesc.add("measurements", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv). options(desc).positional(pdesc).run(), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    if (!vm.count("measurements"))
    {
        std::cerr << "No measurements to run" << std::endl;
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
    std::shared_ptr<uint> h_Input = generateRandomInput(SCAN_MAX_LARGE_ARRAY_SIZE, geom_distr_coeff, srand);
    for (auto it = measurements.begin(); it != measurements.end(); it++)
    {
        if (*it == std::string("scan")){
            openOutputFile(ofs, dest, *it, srand, geom_distr_coeff);
            measureScan(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("codecs")){
            openOutputFile(ofs, dest, *it, srand, geom_distr_coeff);
            measureCodecs(h_Input, ofs);
            ofs.close();
        }
        else if (*it == std::string("separate")){
            std::cerr << "separate kernel measurements not yet implemented" << std::endl;
            return 1;
        }
        else if (*it == std::string("integrated")){
            std::cerr << "integrated kernel measurements not yet implemented" << std::endl;
            return 1;
        }
        else {
            std::cerr << "Unknown measurement: " << *it << std::endl;
            return 1;
        }
    }
}
