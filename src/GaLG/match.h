#ifndef GaLG_match_h
#define GaLG_match_h

#include "inv_table.h"
#include "query.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;
using namespace thrust;


typedef struct {
  u32 count;
  double aggregation;
  u32 id;
} data;

namespace GaLG
{
  int h_offsets_initialized = 0;
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
        device_vector<data>& d_data,
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
        device_vector<int>& d_data,
        int& hash_table_size)
  throw (int);
}

#endif
