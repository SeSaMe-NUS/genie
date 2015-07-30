/*
 * interface.h
 *
 *  Created on: Jul 8, 2015
 *      Author: luanwenhao
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "raw_data.h"
#include "inv_list.h"
#include "inv_table.h"
#include "query.h"
#include "match.h"
#include "topk.h"
#include <vector>

#define GALG_DEFAULT_TOPK 10
#define GALG_DEFAULT_RADIUS 0
#define GALG_DEFAULT_THRESHOLD 4
#define GALG_DEFAULT_HASHTABLE_SIZE 0.6
#define GALG_DEFAULT_WEIGHT 1
#define GALG_DEFAULT_DEVICE 0
#define GALG_DEFAULT_NUM_OF_HOT_DIMS 0
#define GALG_DEFAULT_HOT_DIM_THRESHOLD GALG_DEFAULT_THRESHOLD

namespace GaLG
{
	typedef struct _GaLG_Config{
		int num_of_topk;
		int query_radius;
		int count_threshold;
		float hashtable_size;
		int use_device;
		int dim;
		int num_of_hot_dims;
		int hot_dim_threshold;
		bool use_adaptive_range;
		float selectivity;
		std::vector<std::vector<int> > * data_points;
		std::vector<std::vector<int> > * query_points;
		_GaLG_Config():
			num_of_topk(GALG_DEFAULT_TOPK),
			query_radius(GALG_DEFAULT_RADIUS),
			count_threshold(GALG_DEFAULT_THRESHOLD),
			hashtable_size(GALG_DEFAULT_HASHTABLE_SIZE),
			use_device(GALG_DEFAULT_DEVICE),
			data_points(NULL),
			query_points(NULL),
			dim(0),
			num_of_hot_dims(GALG_DEFAULT_NUM_OF_HOT_DIMS),
			hot_dim_threshold(GALG_DEFAULT_HOT_DIM_THRESHOLD),
			use_adaptive_range(false),
			selectivity(-1.0f)
		{}
	} GaLG_Config;

	void knn_search(std::vector<std::vector<int> >& data_points,
					std::vector<std::vector<int> >& query_points,
					std::vector<int>& result,
					int num_of_topk);

	void knn_search(std::vector<std::vector<int> >& data_points,
					std::vector<std::vector<int> >& query_points,
					std::vector<int>& result,
					int num_of_topk,
					int radius,
					int threshold,
					float hashtable,
					int device);

	void knn_search(inv_table& table,
					std::vector<query>& queries,
					std::vector<int>& result,
					int radius,
					int threshold,
					float hashtable,
					int device);

	void knn_search(std::vector<int>& result,
					GaLG_Config& config);

	void knn_search(inv_table& table,
					std::vector<query>& queries,
					std::vector<int>& h_topk,
					GaLG_Config& config);
}




#endif /* INTERFACE_H_ */
