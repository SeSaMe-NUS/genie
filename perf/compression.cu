/**
 * \brief Performance measurement toolkit for compression of inverted lists in GENIE
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>

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



void runSingleScan(uint *h_Input, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_Input, uint *d_Output, size_t arrayLength)
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
    Logger::log(Logger::INFO, "Scan, Array size: %d, Time: %.3f ms, Throughput: %.3f elements per millisecond"
        , arrayLength, scanTime, (double)arrayLength/scanTime);
}

void measureScan(std::shared_ptr<uint> sp_h_Input)
{
    uint *d_Input, *d_Output;
    uint *h_Input = sp_h_Input.get(), *h_OutputCPU, *h_OutputGPU;

    Logger::log(Logger::INFO,"Allocating and initializing CPU & CUDA arrays...\n");

    h_OutputCPU  = (uint *)malloc(SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    h_OutputGPU  = (uint *)malloc(SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    cudaCheckErrors(cudaMalloc((void **)&d_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));

    // Copy input to GPU
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));

    Logger::log(Logger::INFO,"Measuring scan...\n\n");

    initScan();
    for (int length = 4; length <= (int)SCAN_MAX_SHORT_ARRAY_SIZE; length += 4){
        runSingleScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, length);
    }
    closeScan();

    free(h_OutputCPU);
    free(h_OutputGPU);
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
}



template <class CODEC>
void runSingleCodec(uint *h_Input, uint *h_InputCompr, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_InputCompr,
        uint *d_Output, size_t arrayLength, size_t *d_decomprLength)
{
    CODEC codec;
    Logger::log(Logger::INFO,"\n\nTesting codec...\n\n",codec.name().c_str());

    size_t comprLength = SCAN_MAX_LARGE_ARRAY_SIZE;
    memset(h_InputCompr, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    codec.encodeArray(h_Input, arrayLength, h_InputCompr, comprLength);

    // Copy compressed array to GPU
    cudaCheckErrors(cudaMemcpy(d_InputCompr, h_InputCompr, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    // Clear working memory on both GPU and CPU
    cudaCheckErrors(cudaMemset(d_Output, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));
    cudaCheckErrors(cudaMemset(d_decomprLength, 0, sizeof(size_t)));
    memset(h_OutputCPU, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    
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
    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyDeviceToHost));
    size_t decomprLengthGPU;
    cudaCheckErrors(cudaMemcpy(&decomprLengthGPU, d_decomprLength, sizeof(size_t), cudaMemcpyDeviceToHost));

    // run decompression on CPU
    size_t decomprLengthCPU = SCAN_MAX_LARGE_ARRAY_SIZE;
    codec.decodeArray(h_InputCompr, comprLength, h_OutputCPU, decomprLengthCPU);

    // Check if lenth from CPU and GPU decompression matches the original length
    assert(decomprLengthCPU == arrayLength);
    assert(decomprLengthGPU == arrayLength);

    double decomprTime = getInterval(decomprStart, decomprEnd);
    Logger::log(Logger::INFO, "Codec: %s, Array size: %d, Compressed size: %d, Ratio: %.3f bpi, Time (decompr): %.3f ms, Throughput: %.3f of elements per millisecond",
            codec.name().c_str(), arrayLength, comprLength, 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength),
            decomprTime, (double)arrayLength/decomprTime);
}


void measureCodecs(std::shared_ptr<uint> sp_h_Input)
{
    uint *d_Input, *d_Output;
    size_t *d_decomprLength;
    uint *h_Input = sp_h_Input.get(), *h_InputCompr, *h_OutputCPU, *h_OutputGPU;
    const int MAX_UNCOMPRESSED_LENGTH = 1024;

    Logger::log(Logger::INFO,"Allocating and initializing CPU & CUDA arrays...\n");
    
    h_InputCompr = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    h_OutputCPU  = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    h_OutputGPU  = (uint *)malloc(MAX_UNCOMPRESSED_LENGTH * sizeof(uint));
    cudaCheckErrors(cudaMalloc((void **)&d_Input, MAX_UNCOMPRESSED_LENGTH * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, MAX_UNCOMPRESSED_LENGTH * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_decomprLength, sizeof(size_t)));

    // Copy input to GPU
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceVarintCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceJustCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);

    for (int i = 1; i <= MAX_UNCOMPRESSED_LENGTH; i++)
        runSingleCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);


    free(h_OutputCPU);
    free(h_OutputGPU);
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
    cudaCheckErrors(cudaFree(d_decomprLength));
}

std::ofstream openOutputFile();

int main(int argc, char **argv)
{    
    Logger::log(Logger::INFO,"Running %s \n\n", argv[0]);

    double geom_distr_coeff;
    int srand;
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
            "space separated measurements from: scan, codecs, separate, integrated");

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


    std::shared_ptr<uint> h_Input = generateRandomInput(SCAN_MAX_LARGE_ARRAY_SIZE, geom_distr_coeff, srand);
    for (auto it = measurements.begin(); it != measurements.end(); it++)
    {
        if (*it == std::string("scan")){
            measureScan(h_Input);
        }
        else if (*it == std::string("codecs")){
            measureCodecs(h_Input);
        }
        else {
            std::cerr << "Unknown measurement: " << *it << std::endl;
            return 1;
        }
    }
}
