#ifndef GPUGenie_match_h
#define GPUGenie_match_h

#include <stdint.h>
#include <thrust/device_vector.h>

#include "query.h"
#include "inv_table.h"

using namespace std;
using namespace thrust;

typedef unsigned char u8;
typedef uint32_t u32;
typedef unsigned long long u64;

typedef struct data_
{
	u32 id;
	float aggregation;
} data_t;

namespace GPUGenie
{

int
cal_max_topk(vector<query>& queries);

/**
 * @brief Search the inv_table and save the match
 *        result into d_count and d_aggregation.
 * @details Search the inv_table and save the match
 *          result into d_count and d_aggregation.
 *
 * @param table The inv_table which will be searched.
 * @param queries The quries.
 * @param d_data The output data consisting of count, aggregation
 *               and the index of the data in big table.
 * @param hash_table_size The hash table size.
 *
 * @throw inv_table::not_builded_exception if the table has not been builded.
 * @throw inv_table::not_matched_exception if the query is not querying the given table.
 */
void
match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data,
		int hash_table_size, int max_load, int bitmap_bits,
		device_vector<u32>& d_noiih);
void
match(inv_table& table, vector<query>& queries, device_vector<data_t>& d_data,
		device_vector<u32>& d_bitmap, int hash_table_size, int max_load,
		int bitmap_bits, device_vector<u32>& d_noiih);

void
build_queries(vector<query>& queries, inv_table& table,
		vector<query::dim>& dims, int max_load);

}
#endif
