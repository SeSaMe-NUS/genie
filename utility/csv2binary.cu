#include <iostream>
#include <vector>
#include <mpi.h>

#include "GPUGenie.h"

using namespace GPUGenie;

int main(int argc, char *argv[])
{
	if (3 != argc)
		clog << "Usage: ./csv2binary <csv-filename> <binary-filename>" << endl;

	const char * const csv_filename = argv[1];
	const char * const binary_filename = argv[2];

	GPUGenie_Config config;
	inv_table *table = nullptr;
	vector<vector<int> > data;
	config.data_points = &data;
	read_file(data, csv_filename, -1);
	preprocess_for_knn_csv(config, table);
	inv_table::write(binary_filename, table);
}
