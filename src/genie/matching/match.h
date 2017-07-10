/*! \file match.h
 *  \brief This file includes interfaces of original GENIE match functions.
 *
 */
#ifndef GPUGenie_match_h
#define GPUGenie_match_h

#include <stdint.h>
#include <thrust/device_vector.h>

#include <genie/query/query.h>
#include <genie/table/inv_table.h>
#include "match_common.h"

namespace GPUGenie
{

/*! \fn int cal_max_topk(vector<query>& queries)
 *  \brief Find the maximum topk in query set.
 *
 *  \param queries Query set
 *
 *  The function would get the maximum topk of queries in the query set.
 *  And would use this topk as the global topk for all search process.
 *
 *  \return Maximum topk.
 *
 */
int
cal_max_topk(std::vector<query>& queries);

/*!
 *  \brief Search the inv_table and save the match result into d_count and d_aggregation.
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
void
match(
    inv_table& table,
    std::vector<query>& queries,
    thrust::device_vector<genie::matching::data_t>& d_data,
    thrust::device_vector<u32>& d_bitmap,
    int hash_table_size,
    int max_load,
    int bitmap_bits,
    thrust::device_vector<u32>& d_noiih,
    thrust::device_vector<u32>& d_threshold,
    thrust::device_vector<u32>& d_passCount);

void
match_MT(
    std::vector<inv_table*>& table,
    std::vector<std::vector<query> >& queries,
    std::vector<thrust::device_vector<genie::matching::data_t> >& d_data,
    std::vector<thrust::device_vector<u32> >& d_bitmap,
    std::vector<int>& hash_table_size,
    std::vector<int>& max_load,
    int bitmap_bits,
    std::vector<thrust::device_vector<u32> >& d_noiih,
    std::vector<thrust::device_vector<u32> >& d_threshold,
    std::vector<thrust::device_vector<u32> >& d_passCount,
    size_t start,
    size_t finish);


/*! \fn int build_queries(vector<query>& queries, inv_table& table, vector<query::dim>& dims, int max_load)
 *  \brief Collect all the dims in all queries.
 *
 *  \param queries The query set
 *  \param table The inverted table.
 *  \param dims The container for the resulting query details.
 *  \param max_load The maximum number of posting list items that can be processed by one gpu block
 *
 *  \return The max value of counts of queries in the query set.
 */
int
build_queries(std::vector<query>& queries, inv_table& table, std::vector<query::dim>& dims, int max_load);

}
#endif
