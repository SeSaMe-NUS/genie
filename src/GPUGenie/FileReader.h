/*
 * FileReader.h
 *
 *  Created on: Oct 26, 2015
 *      Author: zhoujingbo
 */

#ifndef FILEREADER_H_
#define FILEREADER_H_

#include "query.h"

#include <vector>
#include <string>

namespace GPUGenie {
	typedef struct _GPUGenie_Query_Data{
		int qid;
		int dim;
		int value;
		float sel;
		float weight;
	} attr_t;

	const int GPUGENIE_QUERY_QID_INDEX = 0;
	const int GPUGENIE_QUERY_DIM_INDEX = 1;
	const int GPUGENIE_QUERY_VALUE_INDEX = 2;
	const int GPUGENIE_QUERY_SELECTIVITY_INDEX =3;
	const int GPUGENIE_QUERY_WEIGHT_INDEX = 4;
	const int GPUGENIE_QUERY_NUM_OF_FIELDS = 5;

	void read_file(std::vector<std::vector<int> >& dest,
					const char* fname,
					int num);
	//Read old format query data: same format as data files
	void read_query(inv_table& table,
					const char* fname,
					std::vector<query>& queries,
					int num_of_queries,
					int num_of_query_dims,
					int radius,
					int topk,
					float selectivity);

	//Read new format query data
	//Sample data format
	//qid dim value selectivity weight
	// 0   0   15     0.04        1
	// 0   1   6      0.04        1
	// ....
	void read_query(std::vector<GPUGenie::attr_t>& data, const char* file_name, int num);

    //convert a csv file to a binary file, the third arg determines write style(override or append)
    void csv2binary(const char* csvfile, const char* binaryfile, bool app_write = false);

    //read from a binary style file
    //_data points to a one-dimensional array, data_len is the number of elements
    //_index points to a one-dimensional array, storing beginning poision of each row in data_len[]
    //row_num equals the size of _index[]
    void read_file(const char* fname, int **_data,
                    unsigned int& item_num, unsigned int **_index,
                    unsigned int& row_num);


} /* namespace GPUGenie */


#endif /* FILEREADER_H_ */
