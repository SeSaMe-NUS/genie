#ifndef GaLG_match_h
#define GaLG_match_h

#include "inv_table.h"
#include "query.h"

#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

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
   * @param d_count The count result.
   * @param d_aggregation The aggregation result.
   * 
   * @throw inv_table::not_builded_exception if the table has not been builded.
   * @throw inv_table::not_matched_exception if the query is not querying the given table.
   */
  void
  match(inv_table& table, vector<query>& queries, device_vector<int>& d_count,
      device_vector<float>& d_aggregation) throw (int);

  void
  match(inv_table& table, query& queries, device_vector<int>& d_count,
      device_vector<float>& d_aggregation) throw (int);
}

#endif
