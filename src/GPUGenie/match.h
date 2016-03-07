/*! \file match.h
 *  \brief This file includes interfaces of match functions.
 *
 */
#ifndef GPUGenie_match_h
#define GPUGenie_match_h

#include <stdint.h>
#include <thrust/device_vector.h>

#include "query.h"
#include "inv_table.h"

using namespace std;
using namespace thrust;

/*! \typedef unsigned char u8
 */
typedef unsigned char u8;
/*! \typedef uint32_t u32
 */
typedef uint32_t u32;
/*! \typedef unsigned long long u64
 */
typedef unsigned long long u64;

/*! \struct data_
 *  \brief This is the entry format of the hash table used in GPU.
 *  	   Will be treated as a 64-bit unsigned integer later.
 */

/*! \typedef struct data_ data_t
 */
typedef struct data_
{
	u32 id;/*!< Index of data point */
	float aggregation;/*!< Count of data point*/
} data_t;


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
cal_max_topk(vector<query>& queries);

/*! \fn void match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data, int hash_table_size, int max_load, int bitmap_bits, device_vector<u32>& d_noiih)
 *  \brief Search the inv_table and save the match
 *        result into d_data, which also serves as the hash table.
 *
 *  \param table The inv_table which will be searched.
 *  \param queries The queries.
 *  \param d_data The output data consisting of count and the index of the data in table.
 *  \param hash_table_size The hash table size.
 *  \param max_load The maximum number of posting list items that can be processed by one gpu block
 *  \param bitmap_bits The threshold for the count heap
 *  \param d_noiih The number of items in each hash table
 */
void
match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data,
		int hash_table_size, int max_load, int bitmap_bits,
		device_vector<u32>& d_noiih);

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
void
match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data,
		device_vector<u32>& d_bitmap, int hash_table_size, int max_load,
		int bitmap_bits, device_vector<u32>& d_noiih,
		device_vector<u32>& d_threshold, device_vector<u32>& d_passCount);


/*! \fn int build_queries(vector<query>& queries, inv_table& table, vector<query::dim>& dims, int max_load)
 *  \brief Collect all the dims in all queries.
 *
 *  \param quereis The query set
 *  \param table The inverted table.
 *  \param dims The container for the resulting query details.
 *  \param max_load The maximum number of posting list items that can be processed by one gpu block
 *
 *  \return The max value of counts of queries in the query set.
 */
int
build_queries(vector<query>& queries, inv_table& table,
		vector<query::dim>& dims, int max_load);

}
#endif
