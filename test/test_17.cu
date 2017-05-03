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
    // assert(comprLength <= arrayLength);

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
    Logger::log(Logger::INFO, "Done parallel GPU decompression");
    Logger::log(Logger::INFO, "Codec: %s", codec.name().c_str());
    Logger::log(Logger::INFO, "Array size: %d, Compressed size: %d, Ratio: %f bpi", arrayLength, comprLength,
            32.0 * static_cast<double>(comprLength) / static_cast<double>(arrayLength));
    Logger::log(Logger::INFO, "Time (decompr): %.4f ms, Throughput: %.2f of elements per millisecond",
            decomprTime, (double)arrayLength/decomprTime);

    return ok;
}

const int SCAN_LENGTHS[] =
        {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
        148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208,
        212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 512, 516, 768, 772,
        1020, 1024, 1028, 1524, 2044, 2048};


const int CODEC_LENGTHS[] =
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
        129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
        177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 
        209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 
        225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 
        241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
        257, 511, 512, 513, 767, 768, 769, 1021, 1022, 1023, 1024};

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
    for (auto it = std::begin(SCAN_LENGTHS); it != std::end(SCAN_LENGTHS); it++){
        ok &= testScan(h_Input, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it);
        assert(ok);
    }
    closeScan();

    printf("\n\nTesting codecs...\n\n");
    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceJustCopyCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);

    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceDeltaCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);

    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceBitPackingCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);

    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceVarintCodec>(h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);

    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceJustCopyCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);

    for (auto it = std::begin(CODEC_LENGTHS); it != std::end(CODEC_LENGTHS); it++)
        ok &= testCodec<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>
                (h_Input, h_InputCompr, h_OutputGPU, h_OutputCPU, d_Input, d_Output, *it, d_decomprLength);
    assert(ok);


    printf("Shutting down...\n");
    cudaCheckErrors(cudaFree(d_Output));
    cudaCheckErrors(cudaFree(d_Input));
    cudaCheckErrors(cudaFree(d_decomprLength));

}
