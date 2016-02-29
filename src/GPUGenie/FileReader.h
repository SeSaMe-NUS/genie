/*! \file FileReader.h
 *  \brief This file declares functions about file operations
 *
 */

#ifndef FILEREADER_H_
#define FILEREADER_H_

#include "query.h"

#include <vector>
using namespace std;
namespace GPUGenie
{
/*! \struct _GPUGenie_Query_Data
 *  \brief This struct is used to construct query in multirange mode
 *
 *  This struct is just one range of a query. A query can have as many range as wanted.
 *  However, this struct is only used in multirange mode.
 */

/*! \typedef struct _GPUGenie_Query_Data attr_t
 */

typedef struct _GPUGenie_Query_Data
{
	int qid;/*!< The query id */
	int dim;/*!< The attribute id */
	int value;/*!< The value on that attribute */
	float sel;/*!< selectivity setting */
	float weight;/*!< Weight onn this range*/
} attr_t;

/*! \var const unsigned int GPUGENIE_QUERY_QID_INDEX 0u
 */
const unsigned int GPUGENIE_QUERY_QID_INDEX = 0u;

/*! \var const unsigned int GPUGENIE_QUERY_DIM_INDEX 1u
 */
const unsigned int GPUGENIE_QUERY_DIM_INDEX = 1u;

/*! \var const unsigned int GPUGENIE_QUERY_VALUE_INDEX 2u
 */
const unsigned int GPUGENIE_QUERY_VALUE_INDEX = 2u;

/*! \var const unsigned int GPUGENIE_QUERY_SELECTIVITY_INDEX 3u
 */
const unsigned int GPUGENIE_QUERY_SELECTIVITY_INDEX = 3u;

/*! \var const unsigned int GPUGENIE_QUERY_WEIGHT_INDEX 4u
 */
const unsigned int GPUGENIE_QUERY_WEIGHT_INDEX = 4u;

/*! \var const unsigned int GPUGENIE_QUERY_NUM_OF_FIELDS 5u
 */
const unsigned int GPUGENIE_QUERY_NUM_OF_FIELDS = 5u;

/*! \fn void read_file(vector<vector<int> >& dest, const char* fname, int num)
 *  \brief This function is called to read data from file
 *
 *  \param dest The data would be stored in this variable after finishing reading a file
 *  \param fname The file name.
 *  \param num The number of rows you want to read from the file
 *
 *  This function can only read data from csv file. If the third  parameter is set to -1,
 *  it means reading all the file. For non-multirange of query, we can use this way to read
 *  queries stored in csv files.
 */
void read_file(vector<vector<int> >& dest, const char* fname, int num);

/*! \fn void read_query(inv_table& table, const char* fname, vector<query>& queries, int numn_of_queries, int num_of_query_dims, int radius, int topk, float selectivity)
 *  \brief
 *
 *  \param table
 *  \param fname File name
 *  \param queries
 *  \param num_of_queries Number of queries
 *  \param radius Radius of queries
 *  \param topk Number of results for one query
 *  \selectivity Selectivity of queries
 */

void read_query(inv_table& table, const char* fname,
		vector<query>& queries, int num_of_queries, int num_of_query_dims,
		int radius, int topk, float selectivity);

/*! \fn void read_query(vector<GPUGenie::attr_t>& data, const char* file_name, int num)
 *  \brief Read new format query in multirange mode.
 *
 *  \param data Query would be stored in this variable.
 *  \param file_name File name.
 *  \param num Number of queries to read from a file.
 *
 *  \verbatim
 *  Sample query format
 *  qid dim value selectivity weight
 *  0   0   15     0.04        1
 *  0   1   6      0.04        1
 *  \endverbatim
 */
void read_query(vector<GPUGenie::attr_t>& data, const char* file_name, int num);

//convert a csv file to a binary file, the third arg determines write style(override or append)
/*! \fn void csv2binary(const char* csvfile, const char* binaryfile, bool app_write = false)
 *  \brief This function can convert a csv file to a binary file
 *
 *  \param csvfile The name of the csv file
 *  \param binaryfile The name of the binary file
 *  \app_write The writing mode. True for appending writing, false for trunc writing
 *
 *  Reading csv files is time-comsuming. If you want to read a csv file for many times, we suggest
 *  you first convert the csv file to a binary file. Reading binary file can save a lot of time. You can
 *  build real-time app based on binary reading and searching
 *
 */
void csv2binary(const char* csvfile, const char* binaryfile, bool app_write = false);

/*! \fn void read_file(const char* fname, int **data, unsigned int& item_num, unsigned int **_index, unsigned int& row_num)
 *  \brief This function provides a way to read data points from given dataset.
 *
 *  \param fname The file name
 *  \param _data points to the address of a one-dimension array, storing all data
 *  \param item_num The number of elements in data array
 *  \param _index points to the address of a one-dimension array, storing starting position of each row in data array
 *  \param row_num Equal to the size of index array
 *
 *  Binary reading is recommended for reading large size of dataset or need to do search many times on a same dataset.
 *  It can save you a lot of time.
 */
void read_file(const char* fname, int **_data, unsigned int& item_num, unsigned int **_index, unsigned int& row_num);

} /* namespace GPUGenie */

#endif /* FILEREADER_H_ */
