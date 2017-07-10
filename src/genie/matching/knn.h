/*! \file knn.h
 *  \brief Collection of knn functions.
 *
 */
#ifndef KNN_H_
#define KNN_H_

#include <vector>
#include <thrust/device_vector.h>

#include <genie/table/inv_table.h>
#include <genie/table/inv_compr_table.h>
#include <genie/query/query.h>

namespace GPUGenie
{
/*! \fn void knn(inv_table& table, vector<query>& queries, device_vector<int>& d_top_indexes,
 *     device_vector<int>& d_top_count, int hash_table_size, int max_load, int bitmap_bits)
 *  \brief Find the top k values in given inv_table.
 *
 *  \param table The given table.
 *  \param queries The queries.
 *  \param d_top_indexes The results' indexes.
 *  \param d_top_count The corresponding count to the results indexes.
 *  \param hash_table_size Size of hash table.
 *  \param max_load The max load for each gpu block
 *  \param bitmap_bits The value of maximum count(threshold) for one data point
 *
 *  This function is basic and called by knn_bijectMap(). This function would carry out the process of match-count and topk selection.
 */
void
knn(inv_table& table,
    std::vector<query>& queries,
    thrust::device_vector<int>& d_top_indexes,
    thrust::device_vector<int>& d_top_count,
    int hash_table_size,
    int max_load,
    int bitmap_bits);

void
knn_MT(
    std::vector<inv_table*>& table,
    std::vector<std::vector<query> >& queries,
    std::vector<thrust::device_vector<int> >& d_top_indexes,
    std::vector<thrust::device_vector<int> >& d_top_count,
    std::vector<int>& hash_table_size,
    std::vector<int>& max_load,
    int bitmap_bits);

/*! \fn void knn_bijectMap(inv_table& table,
vector<query>& queries,
 *  device_vector<int>& d_top_indexes, device_vector<int>& d_top_count, int hash_table_size, int max_load, int bitmap_bits)
 *  \brief Find the top k values in given inv_table.
 *
 *  \param table The given table.
 *  \param queries The queries.
 *  \param d_top_indexes The results' indexes.
 *  \param d_top_count The corresponding count to the results indexes.
 *  \param hash_table_size Size of hash table.
 *  \param max_load The max load for each gpu block
 *  \param bitmap_bits The value of maximum count(threshold) for one data point
 *
 *  This function is called by function in interface.h. This function would call knn() in the end.
 */
void
knn_bijectMap(
    inv_table& table,
    std::vector<query>& queries,
    thrust::device_vector<int>& d_top_indexes,
    thrust::device_vector<int>& d_top_count,
    int hash_table_size,
    int max_load,
    int bitmap_bits);

void
knn_bijectMap_MT(
    std::vector<inv_table*>& table,
    std::vector<std::vector<query> >& queries,
    std::vector<thrust::device_vector<int> >& d_top_indexes,
    std::vector<thrust::device_vector<int> >& d_top_count,
    std::vector<int>& hash_table_size,
    std::vector<int>& max_load,
    int bitmap_bits);
}

#endif //ifndef KNN_H_
