/*! \file heap_count.h
 *  \brief This file implements the function for topk selection
 *  in the final hashtable.
 *
 */

#ifndef _HEAP_COUNT_H
#define _HEAP_COUNT_H

#include <thrust/device_vector.h>
#include <genie/matching/match.h>

namespace genie
{
namespace matching
{

/*! \fn heap_count_topk(thrust::device_vector<data_t>& d_data,
				 thrust::device_vector<data_t>& d_topk,
				 thrust::device_vector<u32>& d_threshold,
				 thrust::device_vector<u32>& d_passCount,
				 int topk,
				 int num_of_queries)
 *  \brief Extract topk items from the data vector
 *
 *  Exactly topk items will be extracted from the %device data vector
 *  to the %device result vector according to the thresholds provided.
 *
 *  Note that d_passCount will not be used in the actual computation
 *  but it might be useful later in validation which is not
 *  implemented yet.
 *
 *  \param d_data The %device data vector, which should be the hash table.
 *  \param d_topk The %device result vector to store the resulting topk items.
 *  \param d_threshold The heap count threshold in a %device vector for each hash table.
 *  \param d_passCount The count of passed items in a %device vector for each hash table.
 *  \param topk The number of top items expected.
 *  \param num_of_queries The number of queries that the hash tables correspond to.
 */
void heap_count_topk(thrust::device_vector<genie::matching::data_t>& d_data,
					 thrust::device_vector<genie::matching::data_t>& d_topk,
					 thrust::device_vector<u32>& d_threshold,
					 thrust::device_vector<u32>& d_passCount,
					 int topk,
					 int num_of_queries);

}
}

#endif
