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
#include <GPUGenie/scan.h>
#include <GPUGenie/Timing.h>
#include <GPUGenie/Logger.h>

bool testScan(uint *h_Input, uint *h_OutputGPU, uint *h_OutputCPU, uint *d_Input, uint *d_Output, size_t arrayLength)
{
    cudaCheckErrors(cudaMemcpy(d_Input, h_Input, MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemset(d_Output, 0, MAX_LARGE_ARRAY_SIZE * sizeof(uint)));
    memset(h_OutputCPU, 0, MAX_LARGE_ARRAY_SIZE * sizeof(uint));
    
    uint64_t scanStart = getTime();

    if (arrayLength <= MAX_SHORT_ARRAY_SIZE)
        scanExclusiveShort(d_Output, d_Input, arrayLength);
    else
        scanExclusiveLarge(d_Output, d_Input, arrayLength);

    uint64_t scanEnd = getTime();

    cudaCheckErrors(cudaMemcpy(h_OutputGPU, d_Output, MAX_LARGE_ARRAY_SIZE * sizeof(uint), cudaMemcpyDeviceToHost));

    scanExclusiveHost(h_OutputCPU, h_Input, arrayLength);

    bool ok = true;
    for (uint i = 0; i < MAX_LARGE_ARRAY_SIZE; i++)
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
        for (uint i = 0; i < std::min((uint) arrayLength + 10, MAX_LARGE_ARRAY_SIZE); i++)
            printf("%d ", h_OutputGPU[i]);
        printf("\n");

        printf("h_OutputCPU: ");
        for (uint i = 0; i < std::min((uint) arrayLength + 10, MAX_LARGE_ARRAY_SIZE); i++)
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
    const uint N = MAX_LARGE_ARRAY_SIZE;

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
    ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, MAX_LARGE_ARRAY_SIZE);


    assert(ok);


    // int globalFlag = 1;
    // size_t szWorkgroup;
    // const int iCycles = 100;
    // printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);

    // for (uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1)
    // {
    //     printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     sdkResetTimer(&hTimer);
    //     sdkStartTimer(&hTimer);

    //     for (int i = 0; i < iCycles; i++)
    //     {
    //         szWorkgroup = scanExclusiveShort(d_Output, d_Input, N / arrayLength, arrayLength);
    //     }

    //     checkCudaErrors(cudaDeviceSynchronize());
    //     sdkStopTimer(&hTimer);
    //     double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

    //     printf("Validating the results...\n");
    //     printf("...reading back GPU results\n");
    //     checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

    //     printf(" ...scanExclusiveHost()\n");
    //     scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

    //     // Compare GPU results with CPU results and accumulate error for this test
    //     printf(" ...comparing the results\n");
    //     int localFlag = 1;

    //     for (uint i = 0; i < N; i++)
    //     {
    //         if (h_OutputCPU[i] != h_OutputGPU[i])
    //         {
    //             localFlag = 0;
    //             break;
    //         }
    //     }

    //     // Log message on individual test result, then accumulate to global flag
    //     printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
    //     globalFlag = globalFlag && localFlag;

    //     // Data log
    //     if (arrayLength == MAX_SHORT_ARRAY_SIZE)
    //     {
    //         printf("\n");
    //         printf("scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
    //                (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
    //         printf("\n");
    //     }
    // }

    // printf("***Running GPU scan for large arrays (%u identical iterations)...\n\n", iCycles);

    // for (uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength <<= 1)
    // {
    //     printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     sdkResetTimer(&hTimer);
    //     sdkStartTimer(&hTimer);

    //     for (int i = 0; i < iCycles; i++)
    //     {
    //         szWorkgroup = scanExclusiveLarge(d_Output, d_Input, N / arrayLength, arrayLength);
    //     }

    //     checkCudaErrors(cudaDeviceSynchronize());
    //     sdkStopTimer(&hTimer);
    //     double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

    //     printf("Validating the results...\n");
    //     printf("...reading back GPU results\n");
    //     checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

    //     printf("...scanExclusiveHost()\n");
    //     scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

    //     // Compare GPU results with CPU results and accumulate error for this test
    //     printf(" ...comparing the results\n");
    //     int localFlag = 1;

    //     for (uint i = 0; i < N; i++)
    //     {
    //         if (h_OutputCPU[i] != h_OutputGPU[i])
    //         {
    //             localFlag = 0;
    //             break;
    //         }
    //     }

    //     // Log message on individual test result, then accumulate to global flag
    //     printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
    //     globalFlag = globalFlag && localFlag;

    //     // Data log
    //     if (arrayLength == MAX_LARGE_ARRAY_SIZE)
    //     {
    //         printf("\n");
    //         printf("scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
    //                (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
    //         printf("\n");
    //     }
    // }


    printf("Shutting down...\n");
    closeScan();
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));


    // pass or fail (cumulative... all tests in the loop)
    // exit(globalFlag ? EXIT_SUCCESS : EXIT_FAILURE);
}
