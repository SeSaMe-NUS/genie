#include <fstream>
#include <algorithm>
#include <functional>

#include "file.h"
#include "global.h"
#include <genie/utility/serialization.h>

using namespace GPUGenie;
using namespace std;

void distgenie::file::ReadData(GPUGenie_Config &config, DistGenieConfig &extra_config, vector<vector<int> > &data, vector<shared_ptr<inv_table>> &tables)
{
	if (0 == g_mpi_rank)
		clog << "Start loading tables" << endl;
	string data_file;
	if (0 == extra_config.data_format) // csv
		for (int i = 0; i < extra_config.num_of_cluster; ++i)
		{
			//clog << "load file " << to_string(i) << endl;
			data_file = extra_config.data_file + "_" + to_string(i) + "_" + to_string(g_mpi_rank) + ".csv";
			read_file(data, data_file.c_str(), -1);
			inv_table * raw_table = nullptr;
			preprocess_for_knn_csv(config, raw_table);
			tables.at(i) = shared_ptr<inv_table>(raw_table, [](inv_table* ptr){delete[] ptr;});
		}
	else if (1 == extra_config.data_format) // binary
		for (int i = 0; i < extra_config.num_of_cluster; ++i)
		{
			//clog << "load binary file " << to_string(i) << endl;
			data_file = extra_config.data_file + "_" + to_string(i) + "_" + to_string(g_mpi_rank) + ".dat";
    		tables.at(i) = genie::util::LoadTable(data_file);
			if (config.save_to_gpu && tables.at(i)->d_inv_p == NULL)
				tables.at(i)->cpy_data_to_gpu();
			tables.at(i)->is_stored_in_gpu = config.save_to_gpu;
		}
	// TODO: handle unknown data format
}

void distgenie::file::GenerateOutput(vector<Result> &results, GPUGenie_Config &config, DistGenieConfig &extra_config)
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
