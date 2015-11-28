/*
 * interface.h
 *
 *  Created on: Jul 8, 2015
 *      Author: luanwenhao
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "../GPUGenie.h"
#include <vector>

#define GPUGENIE_DEFAULT_TOPK 10
#define GPUGENIE_DEFAULT_RADIUS 0
#define GPUGENIE_DEFAULT_THRESHOLD 0
#define GPUGENIE_DEFAULT_HASHTABLE_SIZE 1.0f
#define GPUGENIE_DEFAULT_WEIGHT 1
#define GPUGENIE_DEFAULT_DEVICE 0
#define GPUGENIE_DEFAULT_NUM_OF_HOT_DIMS 0
#define GPUGENIE_DEFAULT_HOT_DIM_THRESHOLD GPUGENIE_DEFAULT_THRESHOLD
#define GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE false
#define GPUGENIE_DEFAULT_SELECTIVITY -1.0f
#define GPUGENIE_DEFAULT_POSTING_LIST_LENGTH 100000
#define GPUGENIE_DEFAULT_LOAD_MULTIPLIER 3.0f
#define GPUGENIE_DEFAULT_USE_LOAD_BALANCE false

namespace GPUGenie
{
	typedef struct _GPUGenie_Config{
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

        int *data;//one way to represent data in 1-D array
        unsigned int *index;// preserve the beginning of each row
        unsigned int item_num;
        unsigned int row_num;

		std::vector<std::vector<int> > * query_points;
		int posting_list_max_length;
		float multiplier;
		bool use_load_balance;
		_GPUGenie_Config():
			num_of_topk(GPUGENIE_DEFAULT_TOPK),
			query_radius(GPUGENIE_DEFAULT_RADIUS),
			count_threshold(GPUGENIE_DEFAULT_THRESHOLD),
			hashtable_size(GPUGENIE_DEFAULT_HASHTABLE_SIZE),
			use_device(GPUGENIE_DEFAULT_DEVICE),
			data_points(NULL),

            data(NULL),
            index(NULL),
            item_num(0),
            row_num(0),

			query_points(NULL),
			dim(0),
			num_of_hot_dims(GPUGENIE_DEFAULT_NUM_OF_HOT_DIMS),
			hot_dim_threshold(GPUGENIE_DEFAULT_HOT_DIM_THRESHOLD),
			use_adaptive_range(GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE),
			selectivity(GPUGENIE_DEFAULT_SELECTIVITY),
			posting_list_max_length(GPUGENIE_DEFAULT_POSTING_LIST_LENGTH),
			multiplier(GPUGENIE_DEFAULT_LOAD_MULTIPLIER),
			use_load_balance(GPUGENIE_DEFAULT_USE_LOAD_BALANCE)
		{}
	} GPUGenie_Config;

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

	/**
	* @brief Search on the inverted index and save the result in result
	* bijectMap means building each ordered pair/keyword is also transformed by a bijection map. (Different from the default method, where the
	* keyword is a combination of dimension and value
	* Previous name: knn_search_tweets()
	*        Please refer to /example/example_tweets.cu to see an example about using it
	*
	*/
	void knn_search_bijectMap(std::vector<int>& result,
				GPUGenie_Config& config);



	/**
	* @brief Search on the inverted index and save the result in result
	*        Please refer to /example/example_sift.cu to see an example about using it
	*
	*/
	void knn_search(std::vector<int>& result,
					GPUGenie_Config& config);

	void knn_search(inv_table& table,
					std::vector<query>& queries,
					std::vector<int>& h_topk,
					GPUGenie_Config& config);

    //to provide the load_table function interface, we can make programs more flexible and more adaptive
    void load_table(inv_table& table, std::vector<std::vector<int> >& data_points ,int max_length);
    void load_query(inv_table& table, std::vector<std::vector<int> >& queries, GPUGenie_Config& config);
    void load_query_bijectMap(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config);
    void load_table_bijectMap(inv_table& table, std::vector<std::vector<int> >& data_points, int max_length);

    //below are corresponding functions woring on binary reading results

    void load_table(inv_table& table, int *data, unsigned int item_num, unsigned int *index, unsigned int row_num,int max_length);
    void load_table_bijectMap(inv_table& table, int *data, unsigned int item_num, unsigned int *index,
                                unsigned int row_num, int max_length);



}




#endif /* INTERFACE_H_ */
