#include "GaLG.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>

using namespace GaLG;
using namespace std;

vector<string> split(string& str, const char* c) {
	char *cstr, *p;
	vector<string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL) {
		res.push_back(p);
		p = strtok(NULL, c);
	}
	delete[] cstr;
	return res;
}

string eraseSpace(string origin) {
	int start = 0;
	while (origin[start] == ' ')
		start++;
	int end = origin.length() - 1;
	while (origin[end] == ' ')
		end--;
	return origin.substr(start, end - start + 1);
}

void read_file(vector<vector<int> >& dest,
		        const char* fname,
		        int num)
{
	string line;
	ifstream ifile(fname);

	dest.clear();

	if (ifile.is_open()) {
		int count = 0;
		while (getline(ifile, line) && (count < num || num < 0)) {
			vector<int> row;
			vector<string> nstring = split(line, ", ");
			int i;
			for(i = 0; i < nstring.size(); ++i){
				int int_value = atoi(eraseSpace(nstring[i]).c_str());
				row.push_back(int_value);
			}
			dest.push_back(row);
			count ++;
		}
		printf("%d rows are read into memory!\n", dest.size());
	}

	ifile.close();
}

int main(int argc, char * argv[])
{
	std::vector<std::vector<int> > queries;
	std::vector<std::vector<int> > data;
	inv_table table;

	//Reading file from the disk. Alternatively, one can simply use vectors generated from other functions
	//Example vectors:
	//Properties: 10 points, 5 dimensions, value range 0-255, -1 for excluded dimensions.
	//|index|dim0|dim1|dim2|dim3|dim4|
	//|0	|2	 |255 |16  |0   |-1  |
	//|1	|10	 |-1  |52  |62  |0   |
	//|...  |... |... |... |... |... |
	//|9	|0   |50  |253 |1   |164 |
	read_file(data, "/media/hd1/home/luanwenhao/TestData2Wenhao/sift/sift_100k.csv", -1);
	read_file(queries, "/media/hd1/home/luanwenhao/TestData2Wenhao/sift/sift_100k.csv", 100);

	/*** Configuration of KNN Search ***/
	GaLG::GaLG_Config config;

	//Data dimension
	config.dim = 128;

	//Points with dim counts lower than threshold will be discarded and not shown in topk.
	//It is implemented as a bitmap filter.
	//Set to 0 to disable the feature.
	config.count_threshold = 0;

	//Hash Table size ratio against data size.
	//Topk items will be generated from the hash table so it must be sufficiently large.
	//Max 1.0f and please set to 1.0f if memory allows.
	//Please take note that reducing hash table size is at your own risk.
	config.hashtable_size = 1.0f;

	//Number of topk items desired for each query. (NOT GUARANTEED!)
	//Some queries may result in fewer than desired topk items.
	config.num_of_topk = 100;

	//Query radius from the data point bucket expanding to upward and downward.
	//Will be overwritten by selectivity if use_adaptive_range is set.
	config.query_radius = 0;

	//Index of the GPU device to be used. If you only have one card, then set to 0.
	config.use_device = 1;

	//Number of hot dimensions with long posting lists to be avoided.
	//Once set to n, top n hot dimensions will be split from the query and submit again
	//at the second stage. Set to 0 to disable the feature.
	//May reduce hash table usage and memory usage.
	//Will increase time consumption.
	config.num_of_hot_dims = 0;

	//Threshold for second stage hot dimension scan. Points with counts lower than threshold
	//will not be processed and they will not be present in the hash table.
	//The value should be larger than count_threshold.
	config.hot_dim_threshold = 0;

	//Set if adaptive range of query is used.
	//Once set with a valid selectivity, the query will be re-scanned to
	//guarantee at least (selectivity * data size) of the data points will be matched
	//for each dimension.
	config.use_adaptive_range = true;

	//The selectivity to be used. Range 0.0f (no other bucket to be matched) to 1.0f (match all buckets).
	config.selectivity = 0.004;

	//The pointer to the vector containing the data.
	config.data_points = &data;

	//The pointer to the vector containing the queries.
	config.query_points = &queries;

	/*** End of Configuration ***/

	/*** NOTE TO DEVELOPERS ***/
	/*
	 The library is still under development. Therefore there might be crashes if
	 the parameters are not set optimally.

	 Optimal settings may help you launch ten times more queries than naive setting,
	 but reducing hash table size may result in unexpected crashes or insufficient
	 topk results. (How can you expect 100 topk results from a hash table of size 50
	 or a threshold of 100 suppose you only have 128 dimensions?) So please be careful
	 if you decide to use non-default settings.

	 The basic & must-have settings are:
	 1. dim
	 2. num_of_topk
	 3. data_points
	 4. query_points
	 And leave the rest to default.

	 Recommended settings:
	 If you want to increase the query selectivity (default is one bucket itself, i.e. radius 0,
	 which is absolutely not enough), you can use either way as follows:
	 A. set radius to a larger value, e.g. 1, 2, 3 and so on. It will expand your search
	 	 upwards and downward by the amount of buckets you set. However, it does not guarantee
	 	 the selectivity since data distribution may not even.
	 B. set the selectivity and turn on use_adaptive_range, e.g. selectivity = 0.01 and
	 	 use_adaptive_range = true. It can guarantee that on each dimension the query will match
	 	 for at least (selectivity * data size) number of data points.
	 Note that approach B will overwrite approach A if both are set.

	 Advanced settings:
	 For num_of_hot_dims and hot_dim_threshold, please contact the author for details.

	 Author: Luan Wenhao
	 Contact: lwhluan@gmail.com
	 Date: 24/08/2015
	                          */
	/*** END OF NOTE ***/

	std::vector<int> result;

	printf("Launching knn functions...\n");
	GaLG::knn_search(result, config);

	for(int i = 0; i < 10; ++i)
	{
		printf("Query %d result is: \n\t", i);
		for (int j = 0; j < 10; ++j)
		{
			printf("%d, ", result[i * 100 + j]);
		}
		printf("\n");
	}
}
