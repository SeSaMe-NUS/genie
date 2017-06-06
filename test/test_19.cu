/**
 * Name: test_17.cu
 * Description: Test of parallel codecs
 *
 */

#undef NDEBUG
 
#include <assert.h>
#include <sstream>
#include <stdio.h>
#include <cuda_runtime.h>

#include <GPUGenie/genie_errors.h>
#include <GPUGenie/Timing.h>
#include <GPUGenie/Logger.h>
#include <GPUGenie/DeviceCompositeCodec.h>
#include <GPUGenie/DeviceSerialCodec.h>
#include <GPUGenie/DeviceBitPackingCodec.h>
#include <GPUGenie/DeviceVarintCodec.h>
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
    Logger::log(Logger::INFO, "Scan, Array size: %d, Time: %.3f ms, Throughput: %.3f elements per millisecond"
        , arrayLength, scanTime, (double)arrayLength/scanTime);

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

    if (comprLength > GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH) {
        // Codecs cannot decompress across mutliple blocks, where each block can only decompress lists of length at most
        // GPUGENIE_CODEC_SERIAL_MAX_UNCOMPR_LENGTH
        Logger::log(Logger::INFO, "Codec: %s, Array size: %d, Compressed size: %d, Ratio: %.3f bpi, Compressed list too long",
            codec.name().c_str(), arrayLength, comprLength, 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength));
        return true;
    }

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
            printf ("mismatch: h_Input[%d]=%u, h_OutputCPU[%d]=%u, h_OutputGPU[%d]=%u\n", i, h_Input[i], i, h_OutputCPU[i], i, h_OutputGPU[i]);
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
    Logger::log(Logger::INFO, "Codec: %s, Array size: %d, Compressed size: %d, Ratio: %.3f bpi, Time (decompr): %.3f ms, Throughput: %.3f of elements per millisecond",
            codec.name().c_str(), arrayLength, comprLength, 32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength),
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

    assert(sizeof(long int) >= 8); // otherwise there may be an overflow in these generated numbers
    for (uint i = 0; i < N; i++)
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

    printf("Allocating and initializing CUDA arrays...\n");
    cudaCheckErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
    cudaCheckErrors(cudaMalloc((void **)&d_decomprLength, sizeof(size_t)));

    printf("Testing scan...\n\n");
    initScan();
    bool ok = true;
    for (int i = 4; i <= (int)SCAN_MAX_SHORT_ARRAY_SIZE; i += 4){
        ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i);
        assert(ok);
    }
    closeScan();

    printf("\n\nTesting codecs...\n\n");
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceCopyMultiblockCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceVarintCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceCopyCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);
    
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);
    
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceDeltaCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);
    
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);
    
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceBitPackingCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);
    
    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);

    for (int i = 1; i <= 1024; i++)
        ok &= testCodec<DeviceSerialCodec<DeviceDeltaCodec,DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, i, d_decomprLength);
    assert(ok);


    printf("Shutting down...\n");
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
    cudaCheckErrors(cudaFree(d_decomprLength));

}
