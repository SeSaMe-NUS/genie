#ifndef DEVICE_CODEC_TEMPLATES_IMPL_HPP_
#define DEVICE_CODEC_TEMPLATES_IMPL_HPP_

template <class CODEC> __global__ void
GPUGenie::g_decodeArrayParallel(uint32_t *d_Input, size_t arrayLength, uint32_t *d_Output, size_t capacity, size_t *d_decomprLength)
{
    CODEC codec;
    assert(blockDim.x == codec.decodeArrayParallel_lengthPerBlock() / codec.decodeArrayParallel_threadLoad());
    assert(gridDim.x <= codec.decodeArrayParallel_maxBlocks());
    assert(capacity <= gridDim.x * blockDim.x * codec.decodeArrayParallel_threadLoad());

    
    __shared__ uint32_t s_Input[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];
    __shared__ uint32_t s_Output[GPUGENIE_SCAN_MAX_SHORT_ARRAY_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < codec.decodeArrayParallel_lengthPerBlock(); i += codec.decodeArrayParallel_threadsPerBlock())
    {
        s_Input[idx + i] = (idx + i < (int)arrayLength) ? d_Input[idx + i] : 0;
        s_Output[idx + i] = 0;
    }

    __syncthreads();
    codec.decodeArrayParallel(s_Input, arrayLength, s_Output, capacity);
    __syncthreads();

    for (int i = 0; i < codec.decodeArrayParallel_lengthPerBlock(); i += codec.decodeArrayParallel_threadsPerBlock())
    {
        d_Output[idx + i] = s_Output[idx + i];
    }

    if (idx == 0 && d_decomprLength != NULL)
            (*d_decomprLength) = capacity;
}


template <class CODEC> void
GPUGenie::decodeArrayParallel(
        int blocks,
        int threads,
        uint32_t *d_Input,
        size_t arrayLength,
        uint32_t *d_Output,
        size_t capacity,
        size_t *d_decomprLength)
{
    GPUGenie::g_decodeArrayParallel<CODEC><<<blocks,threads>>>(d_Input, arrayLength, d_Output, capacity, d_decomprLength);
}

#endif
