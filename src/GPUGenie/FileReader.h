/*
 * FileReader.h
 *
 *  Created on: Oct 26, 2015
 *      Author: zhoujingbo
 */

#ifndef FILEREADER_H_
#define FILEREADER_H_

#include "../GPUGenie.h"

#include <vector>
#include <string>

namespace GPUGenie {

	void read_file(std::vector<std::vector<int> >& dest,
					const char* fname,
					int num);

	void read_query(inv_table& table,
					const char* fname,
					std::vector<query>& queries,
					int num_of_queries,
					int num_of_query_dims,
					int radius,
					int topk,
					float selectivity);


    //convert a csv file to a binaryfile, the third arg determines write style(override or append)
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
