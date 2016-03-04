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
 *  \brief This is the entry of hashtable used in GPU.
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
 *        result into d_count and d_aggregation.
 *
 *  \param table The inv_table which will be searched.
 *  \param queries The quries.
 *  \param d_data The output data consisting of count, aggregation
 *               and the index of the data in big table.
 *  \param hash_table_size The hash table size.
 *  \param max_load The maximum length of posting list that can be processed by one gpu block
 *  \param bitmap_bits The threshold value for one data point
 *  \param d_noiih The number of items in hash table
 *
 *  This functions throw two exceptions that may be caught.
 *  throw inv_table::not_builded_exception if the table has not been builded.
 *  throw inv_table::not_matched_exception if the query is not querying the given table.
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
 *  \param queries The quries.
 *  \param d_data The output data consisting of count, aggregation
 *               and the index of the data in big table.
 *  \param hash_table_size The hash table size.
 *  \param max_load The maximum length of posting list that can be processed by one gpu block
 *  \param bitmap_bits The threshold value for one data point
 *  \param d_noiih The number of items in hash table
 *
 *  This functions throw two exceptions that may be caught.
 *  throw inv_table::not_builded_exception if the table has not been builded.
 *  throw inv_table::not_matched_exception if the query is not querying the given table.
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
 *  \param table The given inverted table.
 *  \param dims The dim set of all dims, to be returned.
 *  \param max_load Maximum length of posting list processed by one gpu block.
 *
 *  \return The max count of queries in the query set.
 */
int
build_queries(vector<query>& queries, inv_table& table,
		vector<query::dim>& dims, int max_load);

}
#endif
