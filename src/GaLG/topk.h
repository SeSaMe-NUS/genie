#ifndef GaLG_topk_h
#define GaLG_topk_h

#include "inv_table.h"
#include "query.h"
#include "match.h"

#include <vector>
#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

namespace GaLG
{
void
topk(inv_table& table, query& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits);

void
topk(inv_table& table, query& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits, int dim);

void
topk(inv_table& table, query& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits, int dim, int num_of_hot_dims,int hot_dim_threshold);

void
topk(inv_table& table, vector<query>& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits);

void
topk(inv_table& table, vector<query>& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits, int dim);

void
topk(inv_table& table, vector<query>& queries,
	      device_vector<int>& d_top_indexes, int hash_table_size, int bitmap_bits, int dim, int num_of_hot_dims, int hot_dim_threshold);
  /**
   * @brief Find the top k values in given inv_table.
   * @details Find the top k values in given inv_table.
   *
   * @param table The given table.
   * @param queries The queries.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(inv_table& table, query& queries,
      device_vector<int>& d_top_indexes);

  /**
   * @brief Find the top k values in given inv_table.
   * @details Find the top k values in given inv_table.
   *
   * @param table The given table.
   * @param queries The queries.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(inv_table& table, vector<query>& queries,
      device_vector<int>& d_top_indexes);

  /**
   * @brief Find the top k values in given device_vector.
   * @details Find the top k values in given device_vector.
   *
   * @param d_search The data vector.
   * @param queries The queries.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(device_vector<int>& d_search, vector<query>& queries,
      device_vector<int>& d_top_indexes);

  /**
   * @brief Find the top k values in given device_vector.
   * @details Find the top k values in given device_vector.
   *
   * @param d_search The data vector.
   * @param queries The queries.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(device_vector<float>& d_search, vector<query>& queries,
      device_vector<int>& d_top_indexes);
  void
  topk(device_vector<data_t>& d_search, vector<query>& queries,
      device_vector<int>& d_top_indexe, float dim);
  void
  topk_tweets(GaLG::inv_table& table,
  		   vector<GaLG::query>& queries,
  		   device_vector<int>& d_top_indexes,
  		   int hash_table_size,
  		   int bitmap_bits,
  		   int dim,
  		   int num_of_hot_dims,
  		   int hot_dim_threshold);
  /**
   * @brief Find the top k values in given device_vector.
   * @details Find the top k values in given device_vector.
   *
   * @param d_search The data vecto.
   * @param d_tops The top k values.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(device_vector<int>& d_search, device_vector<int>& d_tops,
      device_vector<int>& d_top_indexes);

  /**
   * @brief Find the top k values in given device_vector.
   * @details Find the top k values in given device_vector.
   * 
   * @param d_search The data vector
   * @param d_tops The top k values.
   * @param d_top_indexes The results' indexes.
   */
  void
  topk(device_vector<float>& d_search, device_vector<int>& d_tops,
      device_vector<int>& d_top_indexes);
  void
  topk(device_vector<data_t>& d_search, device_vector<int>& d_tops,
      device_vector<int>& d_top_indexes, float dim);
}

#endif
