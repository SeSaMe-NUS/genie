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


int main(int argc, char **argv)
{    
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // findCudaDevice(argc, (const char **)argv);

    uint *d_Input, *d_Output;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    const uint N = SCAN_MAX_LARGE_ARRAY_SIZE;

    printf("Allocating and initializing host arrays...\n");
    h_Input     = (uint *)malloc(N * sizeof(uint));
    h_OutputCPU = (uint *)malloc(N * sizeof(uint));
    h_OutputGPU = (uint *)malloc(N * sizeof(uint));
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


    decodeArrayParallel<DeviceDeltaCodec><<<1,SCAN_THREADBLOCK_SIZE>>> ((uint32_t*)d_Input, (uint32_t*)d_Output, 256);


    printf("Shutting down...\n");
    closeScan();
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));

}
