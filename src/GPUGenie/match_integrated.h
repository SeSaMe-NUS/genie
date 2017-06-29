#ifndef GPUGenie_match_integrated_h
#define GPUGenie_match_integrated_h

#include <stdint.h>
#include <vector>

#include <thrust/device_vector.h>

#include "query.h"
#include "inv_compr_table.h"
#include "match.h"



namespace GPUGenie
{

/*! \fn void match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data, device_vector<u32>& d_bitmap,int hash_table_size,
 *          int max_load, int bitmap_bits, device_vector<u32>& d_noiih, device_vector<u32> d_threshold, device_vector<u32>& d_passCount)
 *  \brief Search the inv_table and save the match
 *        result into d_count and d_aggregation.
 *
 *  \param table The inv_table which will be searched.
 *  \param queries The queries.
 *  \param d_data The output data consisting of count and the index of the data in table.
 *  \param hash_table_size The hash table size.
 *  \param max_load The maximum number of posting list items that can be processed by one gpu block
 *  \param bitmap_bits The threshold for the count heap
 *  \param d_noiih The number of items in hash table
 *  \param d_threshold The container for heap-count thresholds of each query.
 *  \param d_passCount The container for heap-count counts in each buckets of each query.
 */
template <class Codec> void
match_integrated(
        inv_compr_table& table,
        std::vector<query>& queries,
        thrust::device_vector<genie::matching::data_t>& d_data,
        thrust::device_vector<u32>& d_bitmap,
        int hash_table_size,
        int bitmap_bits,
        thrust::device_vector<u32>& d_noiih,
        thrust::device_vector<u32>& d_threshold,
        thrust::device_vector<u32>& d_passCount);

}

#endif
