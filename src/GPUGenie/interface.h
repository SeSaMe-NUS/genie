/*! \file interface.h
 *  \brief This file makes it easier for users to call GPUGenie. Thus, users
 *  can focus on this file.
 */

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include <vector>

#include "../GPUGenie.h"

/*! \def GPUGENIE_DEFAULT_TOPK 10
 */
#define GPUGENIE_DEFAULT_TOPK 10
/*! \def GPUGENIE_DEFAULT_RADIUS 0
 */
#define GPUGENIE_DEFAULT_RADIUS 0
/*! \def GPUGENIE_DEFAULT_THRESHOLD 0
 */
#define GPUGENIE_DEFAULT_THRESHOLD 0
/*! \def GPUGENIE_DEFAULT_HASHTABLE_SIZE 1.0f
 */
#define GPUGENIE_DEFAULT_HASHTABLE_SIZE 1.0f
/*! \def GPUGENIE_DEFAULT_WEIGHT 1
 */
#define GPUGENIE_DEFAULT_WEIGHT 1
/*! \def GPUGENIE_DEFAULT_DEVICE 0
 */
#define GPUGENIE_DEFAULT_DEVICE 0
/*! \def GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE false
 */
#define GPUGENIE_DEFAULT_USE_ADAPTIVE_RANGE false
/*! \def GPUGENIE_DEFAULT_SELECTIVITY -1.0f
 */
#define GPUGENIE_DEFAULT_SELECTIVITY -1.0f
/*! \def GPUGENIE_DEFAULT_POSTING_LIST_LENGTH 100000
 */
#define GPUGENIE_DEFAULT_POSTING_LIST_LENGTH 100000
/*! \def GPUGENIE_DEFAULT_LOAD_MULTIPLIER 3.0f
 */
#define GPUGENIE_DEFAULT_LOAD_MULTIPLIER 3.0f
/*! \def GPUGENIE_DEFAULT_USE_LOAD_BALANCE false
 */
#define GPUGENIE_DEFAULT_USE_LOAD_BALANCE false
/*! \def GPUGENIE_DEFAULT_USE_MULTIRANGE true
 */
#define GPUGENIE_DEFAULT_USE_MULTIRANGE true
/*! \def GPUGENIE_DEFAULT_NUM_OF_QUERIES 0
 */
#define GPUGENIE_DEFAULT_NUM_OF_QUERIES 0

namespace GPUGenie
{

/*! \struct _GPUGenie_Config
 *  \brief Definitions about configurations that can be set by users.
 */

/*! \typedef GPUGenie_Config
 *  \brief struct _GPUGenie_Config
 */
typedef struct _GPUGenie_Config
{
	int num_of_topk;/*!< number of results */
	int query_radius;/*!< enhancement of query */
	int count_threshold;/*!< threshold for count */
	float hashtable_size;/*!< size of hashtable for every query */
	int use_device;/*!< id of GPU to use */
	int dim;/*!< dimensions of data points */
	bool use_adaptive_range;/*!< whether to use adaptive range for query */
	float selectivity;/*!< make sure loaded posting lists are of enough length */
	std::vector<std::vector<int> > * data_points;/*!< data set, for data read from csv fiels */

	int *data; /*!< data set, for data read from binary files */
	unsigned int *index; /*!< one level index for data, separating data points in data array */
	unsigned int item_num;/*!< length of data array*/
	unsigned int row_num;/*!< length of index array*/

	int search_type; /*!< 0 for sift-like data search, 1 for bijectMap data search, 2 for specialized sequence search */
	int data_type; /*!< 0 for csv data; 1 for binary data */
	unsigned int max_data_size; /*!< the max number of data items(rows of data), used for multiload feature */
	bool save_to_gpu; /*!< true for transferring data to gpu and keeping in gpu memory */

	std::vector<std::vector<int> > * query_points;/*!< query set, for non-multirange query */
	std::vector<attr_t> * multirange_query_points;/*!< query set, for multirange query */
	int posting_list_max_length;/*!< maximum length of one posting list, used only under load balance setting*/
	float multiplier;/*!< for calculating how long posting list should be fit into one gpu block, used under load balance setting */
	bool use_load_balance;/*!< whether to use load balance feature */
	bool use_multirange;/*!< whether to use multirange query */
	int num_of_queries;/*!< number of queries in one query set */

    bool use_subsequence_search;/*!< whether to use subsequence search*/

    int data_gram_length;/*!< Length of gram in the construction of gram dataset*/
    float edit_distance_diff;/*!< The given upper-bound of edit distance for search. This value will be multiplied by query and the result would be the distance bound*/

    unsigned int num_of_iteration;/*!< Number of iterations. This parameter is used major in sequence search, to cut off obtained knn.*/

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
					GPUGENIE_DEFAULT_NUM_OF_QUERIES), use_subsequence_search(false),
                    data_gram_length(3), edit_distance_diff(0.1),num_of_iteration(1)
	{
	}
} GPUGenie_Config;







/*! \fn bool preprocess_for_knn_csv(GPUGenie_Config& config, inv_table * &_table)
 *  \brief pre-process for knn search on data set read from a csv file
 *
 *  \param config Settings by users
 *  \param _table Pointer to inv_table object array, which can be managed by users
 *
 *  This function includes the process of transferring a data set read from a csv file to
 *  an inv_table, if user turn on multiload feature , the second parameter will point to an
 *  array of inv_table objects. The inv_table objects contain all the information about
 *  corresponding inverted index structure for given data set
 *
 *  \return true only when no error occurs
 */
bool preprocess_for_knn_csv(GPUGenie_Config& config, inv_table * &_table);

/*! \fn bool preprocess_for_knn_binary(GPUGenie_Config& config, inv_table * &_table)
 *  \brief pre-process for knn search on data set read from a binary file
 *
 *  \param config Settings by users
 *  \param _table Pointer to inv_table object array, which can be managed by users
 *
 *  This function includes the process of transferring a data set read from a binary file to
 *  an inv_table, if user turn on multiload feature , the second parameter will point to an
 *  array of inv_table objects. The inv_table objects contain all the information about
 *  corresponding inverted index structure for given data set
 *
 *  \return true only when no error occurs
 */
bool preprocess_for_knn_binary(GPUGenie_Config& config, inv_table * &_table);


/*! \fn knn_search_after_preprocess(GPUGenie_Config& config, inv_table * &_table,vector<int>& result, vector<int>& result_count)
 *  \brief This function is called when preprocess is done,
 *
 *  \param config Settings by users
 *  \param _table Pointer to inv_table object array. It is the identical one in preprocess
 *  \param result One dimensional vector, storing results, separated by topk number
 *  \param result_count Corresponding to result vector. It stores the count number for each result point
 *
 *  This function handle the rest procedure after preprocess finishes,
 *  Multiload is also handled in this function. The results need to be merged in multiload situation
 */
void knn_search_after_preprocess(GPUGenie_Config& config, inv_table * &_table, vector<int>& result, vector<int>& result_count);

/*! \fn knn_search(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config)
 *  \brief Simplest form for use.
 *
 *  \param result One dimensional vector, storing all results for all queries.
 *  \param result_count Corresponding to result vector. It stores the count number for each result point
 *  \param config Settings by users
 *
 *  This function can help users complete search job in one go. This function would call knn_search_for_binary_data()
 *  or knn_search_for_csv_data depending on file tpye where dataset comes from.
 */
void knn_search(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config);

/*! \fn knn_search(vector<int>& result, GPUGenie_Config& config)
 *  \brief Simply call knn_search(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config).
 *
 *  \param result One dimensional vector, storing all results for all queries.
 *  \param config Settings by users
 *
 *  Result_count is left out.
 */
void knn_search(vector<int>& result, GPUGenie_Config& config);

/*! \fn knn_search(inv_table& table, vector<query>& queries, vector<int>& h_topk, vector<int>& h_topk_count, GPUGenie_Config config)
 *  \brief This is a basic function called by all knn search functions.
 *
 *  \param table One inv_table object
 *  \param queries Query set.
 *  \param h_topk Results for all queries.
 *  \param h_topk_count Corresponding to h_topk. It stores all the count numbers
 *
 *  This function is seldom called by users. It is relatively more basic.
 */
void knn_search(inv_table& table, vector<query>& queries,
		vector<int>& h_topk, vector<int>& h_topk_count, GPUGenie_Config& config);

/*! \fn knn_search_for_binary_data(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config)
 *  \brief knn_search for data read from a binary file
 *
 *  \param result Results for all queries
 *  \param result_count Corresponding count numbers for all results in result vector
 *  \param config Setting by users
 *
 *  If you already know the data is coming from a binary file you can call this function.
 *  This function contains all pre-process and post-process.
 */
void knn_search_for_binary_data(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config);

/*! \fn knn_search_for_csv_data(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config)
 *  \brief knn_search for data read from a csv file
 *
 *  \param result Results for all queries
 *  \param result_count Corresponding count numbers for all results in result vector
 *  \param config Setting by users
 *
 *  If you already know the data is coming from a csv file you can call this function.
 *  This function contains all pre-process and post-process.
 */
void knn_search_for_csv_data(vector<int>& result, vector<int>& result_count, GPUGenie_Config& config);


/*! \fn void load_table(inv_table& table, vector<std::vector<int> >& data_points, GPUGenie_Config& config)
 *  \brief This function constructs the inv_table object for the input dataset of sift data.
 *
 *  \param table The constructed table would be returned in this parameter
 *  \param data_points Vector storing all data points
 *  \param config Settings by users
 */
void load_table(inv_table& table, vector<std::vector<int> >& data_points, GPUGenie_Config& config);

/*! \fn void load_query(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config)
 *  \brief This function constructs the corresponding query structure for a specific inv_table object
 *
 *  \param table The inv_table object
 *  \param queries Constructed queries would be returned by this parameter
 *  \config Settings by users
 *
 *  This function would make a choice between load_query_singlerange and load_query_multirange. The
 *  actual constructing process is finished by one of these two functions.
 */
void load_query(inv_table& table, vector<query>& queries, GPUGenie_Config& config);

/*! \fn void load_query_singlerange(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config)
 *  \brief This function construct query structure for queries of non-multirange
 *
 *  \param table The inv_table object
 *  \param queries Constructed queries would be returned by this parameter
 *  \param config Settings by users
 */
void load_query_singlerange(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config);

/*! \fn void load_query_multirange(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config)
 *  \brief This function constructs query structure for queries of multirange
 *
 *  \param table The inv_table object
 *  \param queries Constructed queries would be returned by this parameter
 *  \param config Settings by users
 */
void load_query_multirange(inv_table& table, std::vector<query>& queries, GPUGenie_Config& config);

/*! \fn void load_table_bijectMap(inv_table& table, vector<vector<int> >& data_points, GPUGenie_Config& config)
 *  \brief This function constructs the inv_table object for dataset consisting of non-sift data point vectors
 *
 *  \param table The constructed table would be returned by this parameter
 *  \param data_points The dataset
 *  \param config settings by users
 *
 *  The input dataset can be short text sets.
 */
void load_table_bijectMap(inv_table& table, vector<vector<int> >& data_points, GPUGenie_Config& config);

/*! \fn void load_table(inv_table& table, int *data, unsigned int item_num, unsigned int *index, unsigned int row_num, GPUGenie_Config& config)
 *  \brief This function constructs the inv_table for dataset from binary files
 *
 *  \param table The constructed table would be returned by this parameter
 *  \param data Array for storing all data points in a one-dimension fashion
 *  \param item_num Length for data array
 *  \param index It stores start position for each individual data point in data array
 *  \param row_num Number of data point
 *  \param config Settings by user
 *
 *  This function is responsible for handling data set from binary files. All
 *  data is sift.
 */
void load_table(inv_table& table, int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num, GPUGenie_Config& config);

/*! \fn void load_table_bijectMap(inv_table& table, int *data, unsigned int item_num, unsigned int *index, unsigned int row_num, GPUGenie_Config& config)
 *  \brief This function constructs the inv_table for dataset from binary files
 *
 *  \param table The constructed table would be returned by this parameter
 *  \param data Array for storing all data points in a one-dimension fashion
 *  \param item_num Length for data array
 *  \param index It stores start position for each individual data point in data array
 *  \param row_num Number of data point
 *  \param config Settings by user
 *
 *  This function is responsible for handling data set from binary files. And
 *  the data is for short-text-like data set.
 */
void load_table_bijectMap(inv_table& table, int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num, GPUGenie_Config& config);



/*! \fn void load_table_sequence(inv_table& table, vector<vector<int> > & data_points, GPUGenie_Config& config)
 *  \brief This function handles construction of inv_table for sequence search.
 *
 *  \param table The inv_table to be constructed.
 *  \param data_points The data set to be searched.
 *  \param config The settings from User.
 */
void load_table_sequence(inv_table& table, vector<vector<int> >& data_points, GPUGenie_Config& config);


/*! \fn void load_query_sequence(inv_table& table, vector<query>& queries, GPUGenie_Config& config)
 *  \brief This function help constructs queries' structure on a specific inv_table.
 *
 *  \param table The specific inv_table for dataset.
 *  \param queries The query set to return.
 *  \param config User settings.
 *
 */
void load_query_sequence(inv_table& table, vector<query>& queries, GPUGenie_Config& config);

/*! \fn void sequence_to_gram(vector<vector<int> >& sequences, vector<vector<int> >& gram_data, int max_value, int gram_length)
 *  \brief This function is used to convert initial sequence data to sequences represented by n-gram data
 *
 *  \param sequences The initial sequences.
 *  \param gram_data The data to represent sequence which is broken into n-grams
 *  \param max_value The range of value that can occur in the original sequences, should start at 0.
 *  \param gram_length The length of one n-gram.
 */
void sequence_to_gram(vector<vector<int> >& sequences, vector<vector<int> >& gram_data, int max_value, int gram_length);

/*! \fn void sequence_reduce_to_ground(vector<vector<int> >& data, vector<vector<int> >& converted_data, int& min_value, int& max_value)
 *  \brief Find the max value and min value of data, and subtract each element by min value.
 *
 *  \param data The data waiting to be processed.
 *  \param converted_data It stores a copy of data, where each element is subtracted by the minimum value.
 *  \param min_value The min value returned.
 *  \param max_value The max value returned.
 */
void sequence_reduce_to_ground(vector<vector<int> >& data, vector<vector<int> >& converted_data, int& min_value, int& max_value);


/*! \fn void reset_device()
 *  \brief clear gpu memory
 *
 *  Every time a kernel finishes, there would be some information remained on GPU.
 *  It can cause the same problem as memory leakage. So we have to clear GPU, if we want to
 *  launch multiple queries in one host process
 */

void reset_device();

/*! \fn void get_rowID_offset(vector<int> &result, vector<int> &resultID, vector<int> &resultOffset, unsigned int shift_bits);
 *  \brief Get rowID and corresponding offset in two vectors
 *
 *  \param result The original result with rowID and offset packed together.
 *  \param resultID All the RowID without offset
 *  \param resultOffset All the offset without rowID
 *  \param shift_bits Bits to shift in order to get rowID
 *
 *  In subsequence search, the RowID and Offset are combined according to shift_bits. This function helps to seperate
 *  rowID and  offset.
 */
void get_rowID_offset(vector<int> &result, vector<int> &resultID, vector<int> &resultOffset, unsigned int shift_bits);


}

#endif /* INTERFACE_H_ */
