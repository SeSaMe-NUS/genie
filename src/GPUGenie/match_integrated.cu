
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "match.h"
#include "Logger.h"
#include "PerfLogger.hpp"
#include "Timing.h"
#include "genie_errors.h"
#include "DeviceCompositeCodec.h"
#include "DeviceBitPackingCodec.h"
#include "DeviceVarintCodec.h"

#include "match_integrated.h"

const size_t MATCH_THREADS_PER_BLOCK = 256;

#define OFFSETS_TABLE_16 {0u,       3949349u, 8984219u, 9805709u,\
                          7732727u, 1046459u, 9883879u, 4889399u,\
                          2914183u, 3503623u, 1734349u, 8860463u,\
                          1326319u, 1613597u, 8604269u, 9647369u}

/**
 * Maximal length the codecs are able to decompress into.
 *
 * GENIE uses fixed 256 threads in its kernels. This implies that a Codec has to have a thread load at least 4 (one
 * thread decompressed into 4 values), otherwise such codec will fail.
 */
#define GPUGENIE_INTEGRATED_KERNEL_SM_SIZE (1024)

typedef u64 T_HASHTABLE;
typedef u32 T_KEY;
typedef u32 T_AGE;


namespace GPUGenie
{

template void match_integrated<DeviceCopyCodec>(inv_compr_table&, std::vector<query>&,
thrust::device_vector<data_t>&, thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&,
thrust::device_vector<u32>&, thrust::device_vector<u32>&);

template void match_integrated<DeviceDeltaCodec>(inv_compr_table&, std::vector<query>&, thrust::device_vector<data_t>&,
thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&, thrust::device_vector<u32>&,
thrust::device_vector<u32>&);

template void match_integrated<DeviceBitPackingCodec>(inv_compr_table&, std::vector<query>&,
thrust::device_vector<data_t>&, thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&,
thrust::device_vector<u32>&, thrust::device_vector<u32>&);

template void match_integrated<DeviceVarintCodec>(inv_compr_table&, std::vector<query>&, thrust::device_vector<data_t>&,
thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&, thrust::device_vector<u32>&,
thrust::device_vector<u32>&);

template void match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>(inv_compr_table&,
std::vector<query>&, thrust::device_vector<data_t>&, thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&,
thrust::device_vector<u32>&, thrust::device_vector<u32>&);

template void match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>(inv_compr_table&,
std::vector<query>&, thrust::device_vector<data_t>&, thrust::device_vector<u32>&, int, int, thrust::device_vector<u32>&,
thrust::device_vector<u32>&, thrust::device_vector<u32>&);


std::map<std::string, IntegratedKernelPtr> initIntegratedKernels()
{

    std::map<std::string, IntegratedKernelPtr> kernels;

    kernels["copy"] = match_integrated<DeviceCopyCodec>;
    kernels["d1"] = match_integrated<DeviceDeltaCodec>;
    kernels["bp32"] = match_integrated<DeviceBitPackingCodec>;
    kernels["varint"] = match_integrated<DeviceVarintCodec>;
    kernels["bp32-copy"] = match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceCopyCodec>>;
    kernels["bp32-varint"] = match_integrated<DeviceCompositeCodec<DeviceBitPackingCodec,DeviceVarintCodec>>;

    return kernels;
}

std::map<std::string, IntegratedKernelPtr> integratedKernels = initIntegratedKernels();


int getBitmapSize(int &in_out_bitmap_bits, u32 in_shift_bits_subsequence, int in_number_of_data_points, int in_queries_size)
{
    if (in_out_bitmap_bits < 0)
            in_out_bitmap_bits = -in_out_bitmap_bits; //for hash_table_size, still let it determine by users currently

    int threshold = in_out_bitmap_bits - 1, bitmap_size = 0;
    if (in_out_bitmap_bits > 1)
    {
        float logresult = std::log2((float) in_out_bitmap_bits);
        in_out_bitmap_bits = (int) logresult;
        if (logresult - in_out_bitmap_bits > 0)
        {
            in_out_bitmap_bits += 1;
        }
        logresult = std::log2((float) in_out_bitmap_bits);
        in_out_bitmap_bits = (int) logresult;
        if (logresult - in_out_bitmap_bits > 0)
        {
            in_out_bitmap_bits += 1;
        }
        in_out_bitmap_bits = pow(2, in_out_bitmap_bits);
        bitmap_size = ((((unsigned int)1<<in_shift_bits_subsequence) * in_number_of_data_points) / (32 / in_out_bitmap_bits) + 1)
                * in_queries_size;
    }
    else
    {
        in_out_bitmap_bits = threshold = 0;
    }

    Logger::log(Logger::DEBUG, "Bitmap bits: %d, threshold:%d, shift_bits_subsequence: %d",
            in_out_bitmap_bits, threshold, in_shift_bits_subsequence);

    return bitmap_size;
}


int build_compressed_queries(vector<query>& queries, inv_compr_table *ctable, vector<query::dim>& dims, int max_load)
{
    assert(ctable->build_status() == inv_table::builded);
    assert(ctable->shift_bits_sequence == 0);

    int max_count = -1;
    for (unsigned int i = 0; i < queries.size(); ++i)
    {
        assert (queries[i].ref_table() == ctable);
        queries[i].build_compressed(max_load);
    
        int prev_size = dims.size();
        queries[i].dump(dims);

        int count = dims.size() - prev_size;

        if(count > max_count)
            max_count = count;
    }

    Logger::log(Logger::DEBUG, " dims size: %d.", dims.size()); 
    Logger::log(Logger::DEBUG, "max_count: %d", max_count);
    return max_count;

}

template <class Codec> __global__ void
match_adaptiveThreshold_integrated(
        int m_size, // number of dimensions, i.e. inv_table::m_size()
        int i_size, // number of instances, i.e. inv_table::m_size() * (1u<<shift_bits_subsequence)
        int hash_table_size, // hash table size
        uint32_t* d_compr_inv, // d_uncompr_inv_p points to the start location of uncompr posting list array in GPU memory
        query::dim* d_dims, // compiled queries (dim structure) with locations into d_uncompr_inv
        T_HASHTABLE* hash_table_list, // data_t struct (id, aggregation) array of size queries.size() * hash_table_size
        u32 * bitmap_list, // of bitmap_size
        int bitmap_bits,
        u32* d_topks, // d_topks set to max_topk for all queries
        u32* d_threshold, //initialized as 1, and increase gradually
        u32* d_passCount, //initialized as 0, count the number of items passing one d_threshold
        u32 num_of_max_count, //number of maximum count per query
        u32 * noiih, // number of integers in a hash table; set to 0 for all queries
        bool * overflow,
        unsigned int shift_bits_subsequence)
{
    assert(MATCH_THREADS_PER_BLOCK == blockDim.x);

    assert(m_size != 0 && i_size != 0);

    query::dim& myb_query = d_dims[blockIdx.x];
    int query_index = myb_query.query;
    u32* my_noiih = &noiih[query_index];
    u32* my_threshold = &d_threshold[query_index];
    u32* my_passCount = &d_passCount[query_index * num_of_max_count];         //
    u32 my_topk = d_topks[query_index];                //for AT

    T_HASHTABLE* hash_table = &hash_table_list[query_index * hash_table_size];
    u32 * bitmap;
    if (bitmap_bits)
        bitmap = &bitmap_list[query_index * (i_size / (32 / bitmap_bits) + 1)];

    assert(myb_query.start_pos < myb_query.end_pos);

    int min = myb_query.start_pos;
    int max = myb_query.end_pos;
    size_t comprLength = max - min;
    int order = myb_query.order;

    Codec codec;
    // check if Codec is compatible with the current list
    assert(max - min <= codec.decodeArrayParallel_maxBlocks() * codec.decodeArrayParallel_lengthPerBlock());
    assert(max - min <= gridDim.x * blockDim.x * codec.decodeArrayParallel_threadLoad());
    assert(blockDim.x == codec.decodeArrayParallel_lengthPerBlock() / codec.decodeArrayParallel_threadLoad());

    __shared__ uint32_t s_comprInv[GPUGENIE_INTEGRATED_KERNEL_SM_SIZE];
    __shared__ uint32_t s_decomprInv[GPUGENIE_INTEGRATED_KERNEL_SM_SIZE];

    int idx = threadIdx.x;
    // Copy the compressed list from global memory into shared memory
    // TODO change to more coalesced access (each thread accesses consecutive 128b value)
    for (int i = 0; i < codec.decodeArrayParallel_lengthPerBlock(); i += codec.decodeArrayParallel_threadsPerBlock())
    {
        s_comprInv[idx + i] = (idx + i < (int)comprLength) ? d_compr_inv[idx + i + min] : 0;
        s_decomprInv[idx + i] = 0;
    }
    // set uncompressed length to maximal length, decomprLength also acts as capacity for the codec
    size_t decomprLength = GPUGENIE_INTEGRATED_KERNEL_SM_SIZE;
    __syncthreads();
    codec.decodeArrayParallel(s_comprInv, comprLength, s_decomprInv, decomprLength);
    __syncthreads();

    // if (idx == 0)
    //     printf("Block %d, query %d, start_pos %d, end_pos %d, comprLength %d, decomprLength %d,\n    compr values [0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x],\n    decompr values [%d,%d,%d,%d,%d,%d,%d,%d,%d,%d] \n",
    //         blockIdx.x, query_index, min, max, (int)comprLength, (int)decomprLength,
    //         s_comprInv[0], s_comprInv[1], s_comprInv[2], s_comprInv[3], s_comprInv[4], 
    //         s_comprInv[5], s_comprInv[6], s_comprInv[7], s_comprInv[8], s_comprInv[9],
    //         s_decomprInv[0], s_decomprInv[1], s_decomprInv[2], s_decomprInv[3], s_decomprInv[4], 
    //         s_decomprInv[5], s_decomprInv[6], s_decomprInv[7], s_decomprInv[8], s_decomprInv[9]);

    assert(decomprLength != 0);

    // Iterate the decompressed posting lists array s_decomprIOnv in blocks of MATCH_THREADS_PER_BLOCK
    // docsIDs, where each thread processes one docID at a time
    for (int i = 0; i < ((int)decomprLength - 1) / MATCH_THREADS_PER_BLOCK + 1; ++i)
    {
        if (idx + i * MATCH_THREADS_PER_BLOCK < (int)decomprLength)
        {
            u32 count = 0; //for AT
            u32 access_id = s_decomprInv[idx + i * MATCH_THREADS_PER_BLOCK];// retrieved docID from posting lists array

            if(shift_bits_subsequence != 0)
            {
                int __offset = access_id & (((unsigned int)1<<shift_bits_subsequence) - 1);
                int __new_offset = __offset - order;
                if(__new_offset >= 0)
                {
                    access_id = access_id - __offset + __new_offset;
                }
                else
                    continue;
            }

            u32 thread_threshold = *my_threshold;
            bool key_eligible;                //
            if (bitmap_bits)
            {

                key_eligible = false;
                //all count are store in the bitmap, and access the count
                count = bitmap_kernel_AT(access_id, bitmap, bitmap_bits,
                        thread_threshold, &key_eligible);

                if (!key_eligible)
                    continue;                //i.e. count< thread_threshold
            }

            key_eligible = false;
            if (count < *my_threshold)
            {
                continue;      //threshold has been increased, no need to insert
            }

            //Try to find the entry in hash tables
            bool pass_threshold;    //to determine whether pass the check of my_theshold
            access_kernel_AT(
                    access_id,               
                    hash_table, hash_table_size, myb_query, count, &key_eligible,
                    my_threshold, &pass_threshold);

            if (key_eligible)
            {
                if (pass_threshold)
                {
                    updateThreshold(my_passCount, my_threshold, my_topk, count);
                }

                continue;
            }

            if (!key_eligible)
            {
                //Insert the key into hash table
                //access_id and its location are packed into a packed key

                if (count < *my_threshold)
                {
                    continue;//threshold has been increased, no need to insert
                }

                hash_kernel_AT(access_id, hash_table, hash_table_size, myb_query, count,
                        my_threshold, my_noiih, overflow, &pass_threshold);
                if (*overflow)
                {

                    return;
                }
                if (pass_threshold)
                {
                    updateThreshold(my_passCount, my_threshold, my_topk, count);
                }
            }

        }
    }
}

template <class Codec> void
match_integrated(
        inv_compr_table& table,
        std::vector<query>& queries,
        thrust::device_vector<data_t>& d_hash_table,
        thrust::device_vector<u32>& d_bitmap,
        int hash_table_size,
        int bitmap_bits,
        thrust::device_vector<u32>& d_num_of_items_in_hashtable,
        thrust::device_vector<u32>& d_threshold,
        thrust::device_vector<u32>& d_passCount)
{
    // GPU time measuring events (kernel executions are measured using CUDA events)
    cudaEvent_t startMatching, stopMatching;
    cudaEvent_t startConvert, stopConvert;
    cudaEventCreate(&startMatching);
    cudaEventCreate(&stopMatching);
    cudaEventCreate(&startConvert);
    cudaEventCreate(&stopConvert);
    // CPU time measuring
    u64 overallStart, overallEnd;
    u64 queryCompilationStart, queryCompilationEnd;
    u64 preprocessingStart, preprocessingEnd;
    u64 queryTransferStart, queryTransferEnd;
    u64 dataTransferStart, dataTransferEnd;
    u64 constantTransferStart, constantTransferEnd;
    u64 allocationStart, allocationEnd;
    u64 fillingStart, fillingEnd;



    Logger::log(Logger::INFO, "*** Starting matching (Integrated Compressed)...");
    overallStart = getTime();

    // Make sure if we decompress a single lists from the table, we can fit it into shared memory
    assert(table.getUncompressedPostingListMaxLength() <= GPUGENIE_INTEGRATED_KERNEL_SM_SIZE);
    assert(table.build_status() == inv_table::builded);



    Logger::log(Logger::INFO, "    Preprocessing variables for matching kernel...");
    preprocessingStart = getTime();

    u32 shift_bits_subsequence = table._shift_bits_subsequence();
    int bitmap_size = getBitmapSize(bitmap_bits, shift_bits_subsequence, table.i_size(), queries.size());
    assert(bitmap_size > 0);
    u32 max_topk = cal_max_topk(queries);

    preprocessingEnd = getTime();



    Logger::log(Logger::INFO, "    Compiling queries...");
    queryCompilationStart = getTime();

    vector<query::dim> dims;        
    //number of maximum count per query
    u32 num_of_max_count = build_compressed_queries(
        queries, &table, dims, table.getUncompressedPostingListMaxLength());

    queryCompilationEnd = getTime();



    Logger::log(Logger::INFO, "    Transferring queries to device...");
    queryTransferStart = getTime();

    thrust::device_vector<query::dim> d_dims(dims);

    queryTransferEnd  = getTime();
    

    
    Logger::log(Logger::INFO, "    Transferring inverted lists to device...");
    dataTransferStart = getTime();

    if (table.get_total_num_of_table() > 1 || !table.is_stored_in_gpu)
        table.cpy_data_to_gpu();

    dataTransferEnd  = getTime();



    Logger::log(Logger::INFO, "    Transferring constant symbol memory to device...");
    constantTransferStart = getTime();

    u32 h_offsets[16] = OFFSETS_TABLE_16;
    cudaCheckErrors(cudaMemcpyToSymbol(GPUGenie::offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));
    Logger::log(Logger::INFO, "        Transferring offsets table (total %d bytes)", sizeof(u32)*16);

    constantTransferEnd = getTime();



    Logger::log(Logger::INFO, "    Allocating matching memory on device...");
    allocationStart = getTime();

    Logger::log(Logger::INFO, "        Allocating threshold (total %d bytes)...", queries.size() * sizeof(u32));
    d_threshold.resize(queries.size());

    Logger::log(Logger::INFO, "        Allocating passCount (total %d bytes)...", queries.size() * num_of_max_count * sizeof(u32));
    d_passCount.resize(queries.size()*num_of_max_count);

    Logger::log(Logger::INFO, "        Allocating bitmap (total %d bytes)...", bitmap_size * sizeof(u32));
    d_bitmap.resize(bitmap_size);
    
    Logger::log(Logger::INFO, "        Allocating num_of_items_in_hashtable (total %d bytes)...", queries.size() * sizeof(u32));
    d_num_of_items_in_hashtable.resize(queries.size());
    
    Logger::log(Logger::INFO, "        Allocating d_topks (total %d bytes)...", queries.size() * sizeof(u32));
    thrust::device_vector<u32> d_topks;
    d_topks.resize(queries.size());

    Logger::log(Logger::INFO, "        Allocating hash_table (total %d bytes)...", queries.size() * hash_table_size * sizeof(data_t));
    d_hash_table.resize(queries.size() * hash_table_size);

    bool h_overflow[1] = {false};
    bool * d_overflow_p;
    Logger::log(Logger::INFO, "        Allocating hash table overflow indicator (total %d bytes)...", sizeof(bool));
    cudaCheckErrors(cudaMalloc((void**) &d_overflow_p, sizeof(bool)));

    allocationEnd = getTime();


    
    Logger::log(Logger::INFO, "    Matching...");
    for (int loop_count = 1; ;loop_count++)
    {
        Logger::log(Logger::INFO, "    Preparing matching... (attempt %d)", loop_count);
        Logger::log(Logger::INFO, "    Filling matching memory on device...");
        fillingStart = getTime();

        thrust::fill(d_threshold.begin(), d_threshold.end(), 1);
        thrust::fill(d_passCount.begin(), d_passCount.end(), 0u);
        thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
        thrust::fill(d_num_of_items_in_hashtable.begin(), d_num_of_items_in_hashtable.end(), 0u);
        thrust::fill(d_topks.begin(), d_topks.end(), max_topk);
        thrust::fill(d_hash_table.begin(), d_hash_table.end(), data_t{0u, 0.0f}); 

        h_overflow[0] = false;
        cudaCheckErrors(cudaMemcpy(d_overflow_p, h_overflow, sizeof(bool), cudaMemcpyHostToDevice));

        fillingEnd = getTime();
        

        Logger::log(Logger::INFO, "    Starting decompression & match kernel...");
        cudaEventRecord(startMatching);
        
        // Call matching kernel, where each BLOCK does matching of one compiled query, only matching for the
        // next DECOMPR_BATCH compiled queries is done in one invocation of the kernel -- this corresponds to
        // the number of decompressed inverted lists
        match_adaptiveThreshold_integrated<Codec><<<dims.size(),MATCH_THREADS_PER_BLOCK>>>
               (table.m_size(),
                (table.i_size() * ((unsigned int)1<<shift_bits_subsequence)),
                hash_table_size, // hash table size
                table.deviceCompressedInv(), // points to the start location of compressed posting list array in GPU mem
                thrust::raw_pointer_cast(d_dims.data()), // compiled queries (dim structure)
                reinterpret_cast<T_HASHTABLE*>(thrust::raw_pointer_cast(d_hash_table.data())),
                thrust::raw_pointer_cast(d_bitmap.data()), // of bitmap_size
                bitmap_bits,
                thrust::raw_pointer_cast(d_topks.data()), // d_topks set to max_topk for all queries
                thrust::raw_pointer_cast(d_threshold.data()),
                thrust::raw_pointer_cast(d_passCount.data()), //initialized as 0, count the number of items passing one d_threshold
                num_of_max_count,//number of maximum count per query
                thrust::raw_pointer_cast(d_num_of_items_in_hashtable.data()), // number of integers in a hash table set to 0 for all queries
                d_overflow_p, // bool
                shift_bits_subsequence);

        cudaEventRecord(stopMatching);
        cudaEventSynchronize(stopMatching);
        cudaCheckErrors(cudaDeviceSynchronize());
        
        Logger::log(Logger::INFO, "    Checking for hash table overflow...");
        cudaCheckErrors(cudaMemcpy(h_overflow, d_overflow_p, sizeof(bool), cudaMemcpyDeviceToHost));
        if(!h_overflow[0]){
            Logger::log(Logger::INFO, "    Matching succeeded");
            break;
        }

        Logger::log(Logger::INFO, "    Matching failed (hash table overflow)");
        hash_table_size += num_of_max_count*max_topk;
        if(hash_table_size > table.i_size())
            hash_table_size = table.i_size();
        
        d_hash_table.resize(queries.size()*hash_table_size);
        Logger::log(Logger::INFO, "    Resized hash table (now total of %d bytes)",
            queries.size() * hash_table_size * sizeof(data_t));
    };

    Logger::log(Logger::INFO, "    Starting data conversion from hash tables......");
    Logger::log(Logger::INFO, "    Starting conversion kernel...");
    cudaEventRecord(startConvert);

    convert_to_data<<<hash_table_size*queries.size() / 1024 + 1,1024>>>(
        reinterpret_cast<T_HASHTABLE*>(thrust::raw_pointer_cast(d_hash_table.data())),
        (u32)hash_table_size*queries.size());

    cudaEventRecord(stopConvert);
    cudaCheckErrors(cudaEventSynchronize(stopConvert));


    // Only deallocate manually allocated memory; thrust::device_vector will be deallocated when out of scope 
    Logger::log(Logger::INFO, "    Deallocating memory......");
    cudaCheckErrors(cudaFree(d_overflow_p));

    Logger::log(Logger::INFO, "    Matching is done!");
    overallEnd = getTime();


    float matchingTime, convertTime;
    cudaEventElapsedTime(&matchingTime, startMatching, stopMatching);
    cudaEventElapsedTime(&convertTime, startConvert, stopConvert);

    Codec c;
    PerfLogger::get().ofs()
        << c.name() << ","
        << std::fixed << std::setprecision(3) << getInterval(overallStart, overallEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(queryCompilationStart, queryCompilationEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(preprocessingStart, preprocessingEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(queryTransferStart, queryTransferEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(dataTransferStart, dataTransferEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(constantTransferStart, constantTransferEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(allocationStart, allocationEnd) << ","
        << std::fixed << std::setprecision(3) << getInterval(fillingStart, fillingEnd) << ","
        << std::fixed << std::setprecision(3) << matchingTime << ","
        << std::fixed << std::setprecision(3) << convertTime << std::endl;

}

}
