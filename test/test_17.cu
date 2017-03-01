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
        uint *d_Output, size_t arrayLength)
{
    CODEC bpCodec;

    size_t comprLength = SCAN_MAX_LARGE_ARRAY_SIZE;
    memset(h_InputCompr, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    bpCodec.encodeArray(h_Input, arrayLength, h_InputCompr, comprLength);
    assert(comprLength <= arrayLength);

    printf("h_Input: ");
    for (uint i = 0; i < std::min((uint) arrayLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
        printf("%d ", h_Input[i]);
    printf("\n");

    printf("h_InputCompr: ");
    for (uint i = 0; i < std::min((uint) comprLength + 10, SCAN_MAX_LARGE_ARRAY_SIZE); i++)
        printf("0x%08X ", h_InputCompr[i]);
    printf("\n");

    cudaCheckErrors(cudaMemcpy(d_InputCompr, h_InputCompr, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemset(d_Output, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint)));
    memset(h_OutputCPU, 0, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    
    uint64_t decomprStart = getTime();

    decodeArrayParallel<CODEC><<<1,SCAN_THREADBLOCK_SIZE>>>(d_InputCompr, d_Output, comprLength, SCAN_MAX_LARGE_ARRAY_SIZE);
    
    uint64_t decomprEnd = getTime();

    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE * sizeof(uint),
        cudaMemcpyDeviceToHost));

    size_t decomprLength = SCAN_MAX_LARGE_ARRAY_SIZE;
    bpCodec.decodeArray(h_OutputCPU, comprLength, h_InputCompr, decomprLength);

    // Check if lenth from CPU decompression matches the original length
    assert(decomprLength == arrayLength);

    // Compare original array with CPU decompressed array and GPU decompressed array
    bool ok = true;
    for (uint i = 0; i < SCAN_MAX_LARGE_ARRAY_SIZE; i++)
    {
        if (h_OutputCPU[i] != h_OutputGPU[i] || h_OutputGPU[i] != h_Input[i])
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

    double decomprTime = getInterval(decomprStart, decomprEnd);
    Logger::log(Logger::INFO, "Decompression on array of size %d took %.4ff ms. Throughput: %.4f milions of elements per second"
        , arrayLength, decomprTime, 1.0e-6 * (double)arrayLength/decomprTime);

    return ok;
}


int main(int argc, char **argv)
{    
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // findCudaDevice(argc, (const char **)argv);

    uint *d_Input, *d_Output;
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

    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    bool ok = true;
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 256);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 332);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1024);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 1524);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 2048);
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, SCAN_MAX_LARGE_ARRAY_SIZE);
    assert(ok);


    ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, 8);
    assert(ok);

    printf("Shutting down...\n");
    closeScan();
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));

}
