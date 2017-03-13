/**
 * Name: test_17.cu
 * Description: Test of parallel codecs
 *
 */

#include <assert.h>
#include <sstream>
#include <stdio.h>
#include <cuda_runtime.h>

#include <GPUGenie/genie_errors.h>
#include <GPUGenie/Timing.h>
#include <GPUGenie/Logger.h>
#include <GPUGenie/DeviceCompositeCodec.h>
#include <GPUGenie/DeviceBitPackingCodec.h>
#include <GPUGenie/scan.h> 

using namespace GPUGenie;

bool testScan(uint *h_Input, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_Input, uint *d_Output, size_t arrayLength)
{
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemset(d_Output, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));

    memset(h_OutputCPU, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    
    uint64_t scanStart = getTime();

    if (arrayLength <= SCAN_MAX_SHORT_ARRAY_SIZE)
        scanExclusiveShort(d_Output, d_Input, arrayLength);
    else
        scanExclusiveLarge(d_Output, d_Input, arrayLength);

    uint64_t scanEnd = getTime();

    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint),
        cudaMemcpyDeviceToHost));

    scanExclusiveHost(h_OutputCPU, h_Input, arrayLength);

    bool ok = true;
    for (uint i = 0; i < SCAN_MAX_LARGE_ARRAY_SIZE; i++)
    {
        if (h_OutputCPU[i] != h_OutputGPU[i])
        {
            ok = false;
            break;
        }
    }
    if (!ok)
    {
        printf("h_OutputGPU: ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
            printf("%d ", h_OutputGPU[i]);
        printf("\n");

        printf("h_OutputCPU: ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
            printf("%d ", h_OutputCPU[i]);
        printf("\n");
    }

    double scanTime = getInterval(scanStart, scanEnd);
    Logger::log(Logger::INFO, "Scan on array of size %d took %.4ff ms. Throughput: %.4f milions of elements per second"
        , arrayLength, scanTime, 1.0e-6 * (double)arrayLength/scanTime);

    return ok;
}

template <class CODEC>
bool testCodec(uint *h_Input, uint *h_InputCompr, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_InputCompr,
        uint *d_Output, size_t arrayLength, size_t *d_decomprLength)
{
    CODEC codec;

    size_t comprLength = SCAN_MAX_LARGE_ARRAY_SIZE;
    memset(h_InputCompr, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    codec.encodeArray(h_Input, arrayLength, h_InputCompr, comprLength);
    assert(comprLength <= arrayLength);

    // printf("codec.name(): %s\n", codec.name().c_str());
    // printf("arrayLength: %lu\n", arrayLength);
    // printf("comprLength: %lu\n", comprLength);

    // printf("h_Input: ");
    // for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
    //     printf("%d ", h_Input[i]);
    // printf("\n");

    // printf("h_InputCompr: ");
    // for (uint i = 0; i < std::min((uint) comprLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
    //     printf("0x%08X ", h_InputCompr[i]);
    // printf("\n");

    cudaCheckErrors(cudaMemcpy(d_InputCompr, h_InputCompr, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
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

    // Compare original array with CPU decompressed array and GPU decompressed array
    bool ok = true;
    for (int i = 0; i < (int)arrayLength; i++)
    {
        if (h_OutputCPU[i] != h_OutputGPU[i] || h_OutputGPU[i] != h_Input[i])
        {
            ok = false;
            printf ("mismatch: h_Input[%d]=%u, h_OutputCPU[%d]=%u, h_Input[%d]=%u\n", i, h_Input[i], i, h_OutputCPU[i], i, h_OutputGPU[i]);
            break;
        }
    }
    if (!ok)
    {
        printf("h_Input:     ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
            printf("%u ", h_Input[i]);
        printf("\n");

        printf("h_OutputGPU: ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
            printf("%u ", h_OutputGPU[i]);
        printf("\n");

        printf("h_OutputCPU: ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
            printf("%u ", h_OutputCPU[i]);
        printf("\n");
    }

    double decomprTime = getInterval(decomprStart, decomprEnd);
    Logger::log(Logger::INFO, "Done parallel GPU decompression");
    Logger::log(Logger::INFO, "Codec: %s", codec.name().c_str());
    Logger::log(Logger::INFO, "Array size: %d, Compressed size: %d, Ratio: %f bpi", arrayLength, comprLength,
            32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength));
    Logger::log(Logger::INFO, "Time (decompr): %.4f ms, Throughput: %.2f of elements per second",
            decomprTime, (double)arrayLength/decomprTime);

    return ok;
}


int main(int argc, char **argv)
{    
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // findCudaDevice(argc, (const char **)argv);

    uint *d_Input, *d_Output;
    size_t *d_decomprLength;
    uint *h_Input, *h_InputCompr, *h_OutputCPU, *h_OutputGPU;
    const uint N = SCAN_MAX_LARGE_ARRAY_SIZE;

    printf("Allocating and initializing host arrays...\n");
    h_Input      = (uint *)malloc(N * sizeof(uint));
    h_InputCompr = (uint *)malloc(N * sizeof(uint));
    h_OutputCPU  = (uint *)malloc(N * sizeof(uint));
    h_OutputGPU  = (uint *)malloc(N * sizeof(uint));
    srand(666);

    for (uint i = 0; i < N; i++)
        h_Input[i] = rand() % 100;

    printf("Allocating and initializing CUDA arrays...\n");
    cudaCheckErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_decomprLength, sizeof(size_t)));

    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    bool ok = true;
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 4);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1524);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 2048);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE);
    assert(ok);

    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1, d_decomprLength);
    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8, d_decomprLength);
    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256, d_decomprLength);
    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332, d_decomprLength);
    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024, d_decomprLength);
    assert(ok);

    ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1, d_decomprLength);
    ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8, d_decomprLength);
    ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256, d_decomprLength);
    ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332, d_decomprLength);
    ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024, d_decomprLength);
    assert(ok);

    ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1, d_decomprLength);
    ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8, d_decomprLength);
    ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256, d_decomprLength);
    ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332, d_decomprLength);
    ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024, d_decomprLength);
    assert(ok);

    ok &= testCodec<DeviceBitPackingPrefixedCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1, d_decomprLength);
    ok &= testCodec<DeviceBitPackingPrefixedCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8, d_decomprLength);
    ok &= testCodec<DeviceBitPackingPrefixedCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256, d_decomprLength);
    ok &= testCodec<DeviceBitPackingPrefixedCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332, d_decomprLength);
    ok &= testCodec<DeviceBitPackingPrefixedCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024, d_decomprLength);
    assert(ok);


    printf("Shutting down...\n");
    closeScan();
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
    cudaCheckErrors(cudaFree(d_decomprLength));

}
