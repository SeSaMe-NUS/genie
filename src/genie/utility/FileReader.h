/*! \file FileReader.h
 *  \brief This file declares functions about file operations
 *
 */

#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <vector>

namespace genie {namespace table {class inv_table; }}
namespace genie {namespace query {class Query; }}

namespace genie
{
namespace utility
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

/*! \var const unsigned int GPUGENIE_QUERY_QID_INDEX
 */
const unsigned int GPUGENIE_QUERY_QID_INDEX = 0u;

/*! \var const unsigned int GPUGENIE_QUERY_DIM_INDEX
 */
const unsigned int GPUGENIE_QUERY_DIM_INDEX = 1u;

/*! \var const unsigned int GPUGENIE_QUERY_VALUE_INDEX
 */
const unsigned int GPUGENIE_QUERY_VALUE_INDEX = 2u;

/*! \var const unsigned int GPUGENIE_QUERY_SELECTIVITY_INDEX
 */
const unsigned int GPUGENIE_QUERY_SELECTIVITY_INDEX = 3u;

/*! \var const unsigned int GPUGENIE_QUERY_WEIGHT_INDEX
 */
const unsigned int GPUGENIE_QUERY_WEIGHT_INDEX = 4u;

/*! \var const unsigned int GPUGENIE_QUERY_NUM_OF_FIELDS
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
void read_file(std::vector<std::vector<int> >& dest, const char* fname, int num);

/*! \fn void read_query(inv_table& table, const char* fname, vector<query>& queries, int num_of_queries, int num_of_query_dims, int radius, int topk, float selectivity)
 *  \brief Process queries, read from file.
 *
 *  \param table The given table
 *  \param fname File name
 *  \param queries Query set
 *  \param num_of_queries Number of queries
 *  \param num_of_query_dims Number of dim structs of queries.
 *  \param radius Radius of queries
 *  \param topk Number of results for one query
 *  \param selectivity Selectivity of queries
 */

void read_query(
        genie::table::inv_table &table, const char* fname, std::vector<genie::query::Query> &queries,
        int num_of_queries, int num_of_query_dims, int radius, int topk, float selectivity);

/*! \fn void read_query(vector<attr_t>& data, const char* file_name, int num)
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
void read_query(std::vector<attr_t>& data, const char* file_name, int num);

} // namespace utility
} // namesapce genie

#endif /* FILEREADER_H_ */
