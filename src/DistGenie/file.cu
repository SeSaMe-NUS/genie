#include <omp.h>
#include <fstream>
#include <algorithm>
#include <functional>

#include "file.h"
#include "global.h"

using namespace GPUGenie;
using namespace std;
using namespace DistGenie;

void DistGenie::ReadData(GPUGenie_Config &config, ExtraConfig &extra_config, vector<vector<int> > &data, inv_table **tables)
{
	string data_file;
	if (0 == extra_config.data_format) // csv
//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < extra_config.num_of_cluster; ++i)
		{
			clog << "load file " << to_string(i) << endl;
			data_file = extra_config.data_file + "_" + to_string(i) + "_" + to_string(g_mpi_rank) + ".csv";
			read_file(data, data_file.c_str(), -1);
			preprocess_for_knn_csv(config, tables[i]);
		}
	else if (1 == extra_config.data_format) // binary
//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < extra_config.num_of_cluster; ++i)
		{
			clog << "load binary file " << to_string(i) << endl;
			data_file = extra_config.data_file + "_" + to_string(i) + "_" + to_string(g_mpi_rank) + ".dat";
			inv_table::read(data_file.c_str(), tables[i]);
			tables[i]->cpy_data_to_gpu();
		}
	// TODO: handle unknown data format
}

void DistGenie::GenerateOutput(vector<Result> &results, GPUGenie_Config &config, ExtraConfig &extra_config)
{
	int topk = config.num_of_topk;
	ofstream output(extra_config.output_file);
	for (auto it = results.begin(); it != results.end(); ++it)
	{
		sort(it->begin(), it->end(), std::greater<std::pair<int, int> >());
		for (int i = 0; i < topk; ++i)
			output << it->at(i).second << "," << it->at(i).first << endl;
	}
	output.close();
}
