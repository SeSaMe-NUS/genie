/*
 * interface.h
 *
 *  Created on: Jul 8, 2015
 *      Author: luanwenhao
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include <vector>

#include "../GPUGenie.h"

#define GPUGENIE_DEFAULT_TOPK 10
#define GPUGENIE_DEFAULT_RADIUS 0
#define GPUGENIE_DEFAULT_THRESHOLD 0
#define GPUGENIE_DEFAULT_HASHTABLE_SIZE 1.0f
#define GPUGENIE_DEFAULT_WEIGHT 1
#define GPUGENIE_DEFAULT_DEVICE 0
#define GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE false
#define GPUGENIE_DEFAULT_SELECTIVITY -1.0f
#define GPUGENIE_DEFAULT_POSTING_LIST_LENGTH 100000
#define GPUGENIE_DEFAULT_LOAD_MULTIPLIER 3.0f
#define GPUGENIE_DEFAULT_USE_LOAD_BALANCE false
#define GPUGENIE_DEFAULT_USE_MULTIRANGE true
#define GPUGENIE_DEFAULT_NUM_OF_QUERIES 0

namespace GPUGenie
{
typedef struct _GPUGenie_Config
{
	int num_of_topk;
	int query_radius;
	int count_threshold;
	float hashtable_size;
	int use_device;
	int dim;
	bool use_adaptive_range;
	float selectivity;
	std::vector<std::vector<int> > * data_points;

	int *data; //one way to represent data in 1-D array
	unsigned int *index; // preserve the beginning of each row
	unsigned int item_num;
	unsigned int row_num;

	int search_type; //0 for knn_search, 1 for knn_search_bijectMap
	int data_type; //0 for csv data; 0 for bianry data
	unsigned int max_data_size; // the max number of data items(rows of data)
	bool save_to_gpu; // specify use of multi table

	std::vector<std::vector<int> > * query_points;
	std::vector<attr_t> * multirange_query_points;
	int posting_list_max_length;
	float multiplier;
	bool use_load_balance;
	bool use_multirange;

	int num_of_queries;
	_GPUGenie_Config() :
			num_of_topk(GPUGENIE_DEFAULT_TOPK), query_radius(
					GPUGENIE_DEFAULT_RADIUS), count_threshold(
					GPUGENIE_DEFAULT_THRESHOLD), hashtable_size(
					GPUGENIE_DEFAULT_HASHTABLE_SIZE), use_device(
					GPUGENIE_DEFAULT_DEVICE), data_points(NULL),

			data(NULL), index(NULL), item_num(0), row_num(0),

			search_type(0), data_type(0), max_data_size(0), save_to_gpu(false),

			query_points(NULL), multirange_query_points(NULL), dim(0), use_adaptive_range(
					GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE), selectivity(
					GPUGENIE_DEFAULT_SELECTIVITY), posting_list_max_length(
					GPUGENIE_DEFAULT_POSTING_LIST_LENGTH), multiplier(
					GPUGENIE_DEFAULT_LOAD_MULTIPLIER), use_load_balance(
					GPUGENIE_DEFAULT_USE_LOAD_BALANCE), use_multirange(
					GPUGENIE_DEFAULT_USE_MULTIRANGE), num_of_queries(
					GPUGENIE_DEFAULT_NUM_OF_QUERIES)
	{
	}
} GPUGenie_Config;

/**
 * @brief Search on the inverted index and save the result in result
 * bijectMap means building each ordered pair/keyword is also transformed by a bijection map. (Different from the default method, where the
 * keyword is a combination of dimension and value
 * Previous name: knn_search_tweets()
 *        Please refer to /example/example_tweets.cu to see an example about using it
 *
 */

/**
 * @brief Search on the inverted index and save the result in result
 *        Please refer to /example/example_sift.cu to see an example about using it
 *
 */

bool preprocess_for_knn_csv(GPUGenie_Config& config, inv_table &table,
		inv_table * &_table, unsigned int& table_num);

bool preprocess_for_knn_binary(GPUGenie_Config& config, inv_table& table,
		inv_table * &_table, unsigned int& table_num);

void knn_search_after_preprocess(GPUGenie_Config& config, inv_table& table,
		inv_table * &_table, std::vector<int>& result,
		std::vector<int>& result_count, unsigned int& table_num);

void knn_search(std::vector<int>& result, std::vector<int>& result_count,
		GPUGenie_Config& config);

void knn_search(inv_table& table, std::vector<query>& queries,
		std::vector<int>& h_topk, std::vector<int>& h_topk_count,
		GPUGenie_Config& config);

//For backward compatibility: result_count not included in parameters
void knn_search(std::vector<int>& result, GPUGenie_Config& config);
void knn_search(inv_table& table, std::vector<query>& queries,
		std::vector<int>& h_topk, GPUGenie_Config& config);

void knn_search_for_binary_data(std::vector<int>& result,
		std::vector<int>& result_count, GPUGenie_Config& config);

void knn_search_for_csv_data(std::vector<int>& result,
		std::vector<int>& result_count, GPUGenie_Config& config);

//to provide the load_table function interface, we can make programs more flexible and more adaptive
void load_table(inv_table& table, std::vector<std::vector<int> >& data_points,
		GPUGenie_Config& config);
void load_query(inv_table& table, std::vector<query>& queries,
		GPUGenie_Config& config);
void load_query_singlerange(inv_table& table, std::vector<query>& queries,
		GPUGenie_Config& config);
void load_query_multirange(inv_table& table, std::vector<query>& queries,
		GPUGenie_Config& config);
void load_table_bijectMap(inv_table& table,
		std::vector<std::vector<int> >& data_points, GPUGenie_Config& config);

//below are corresponding functions woring on binary reading results
void load_table(inv_table& table, int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num, GPUGenie_Config& config);
void load_table_bijectMap(inv_table& table, int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num, GPUGenie_Config& config);

}

#endif /* INTERFACE_H_ */
