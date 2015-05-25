#ifndef GaLG_match_h
#define GaLG_match_h

#include "inv_table.h"
#include "query.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;
using namespace thrust;



typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct data_{
  u32 count;
  float aggregation;
  u32 id;
} data_t;

u64 getTime();
float getInterval(u64 t1, u64 t2);
namespace GaLG
{

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
  match(inv_table& table,
        vector<query>& queries,
        device_vector<data_t>& d_data,
        int& hash_table_size) throw (int);

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
  match(inv_table& table,
        query& queries,
        device_vector<data_t>& d_data,
        int& hash_table_size)
  throw (int);
}

#endif
