#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <map>
#include <utility>
#include <algorithm>
#include "GPUGenie.h"

#define LOCAL_RANK atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
#define MPI_DEBUG ">>>MPI DEBUG<<< "
#define pii std::pair<int, int>

using namespace std;
using namespace GPUGenie;

struct ExtraConfig {
	string data_file;
	string query_file;
	string output_file;
};

void ParseConfigurationFile(GPUGenie_Config &, ExtraConfig &, const string);
bool ValidateConfiguration(map<string, string>);

int main(int argc, char *argv[])
{
	/*
	 * MPI initialization
	 */
	int MPI_rank, MPI_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
	if (argc != 2)
	{
		if (MPI_rank == 0)
			cout << "Usage: mpirun -np <proc> --hostfile <hosts> ./dgenie config.file" << endl;
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	/*
	 * read in the config
	 */
	vector<vector<int> > queries;
	vector<vector<int> > data;

	GPUGenie_Config config;
	ExtraConfig extra_config;
	config.query_points = &queries;
	config.data_points = &data;
	string config_filename(argv[1]);
	ParseConfigurationFile(config, extra_config, config_filename);
	cout << MPI_DEBUG << "rank: " << MPI_rank << " using GPU " << config.use_device << endl;

	/*
	 * load data
	 */
	string data_file = extra_config.data_file + "_" + to_string(MPI_rank) + ".csv"; // each process load a different file
	string query_file = extra_config.query_file;
	read_file(data, data_file.c_str(), -1);
	read_file(queries, query_file.c_str(), config.num_of_queries);

	/*
	 * run the queries
	 */
	inv_table *table = NULL;
	vector<int> result;
	vector<int> result_count;
	preprocess_for_knn_csv(config, table);
	knn_search_after_preprocess(config, table, result, result_count);

	/*
	 * merge results from all ranks
	 */
	int *final_result = NULL;       // only for MPI
	int *final_result_count = NULL; // only for MPI
	int single_rank_result_size = config.num_of_topk * config.num_of_queries;
	if (MPI_rank == 0) {
		int result_size = MPI_size * single_rank_result_size;
		final_result = (int *)malloc(sizeof(int) * result_size);
		final_result_count = (int *)malloc(sizeof(int) * result_size);
	}
	MPI_Gather(&result[0], single_rank_result_size, MPI_INT, final_result, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&result_count[0], single_rank_result_size, MPI_INT, final_result_count, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);

	if (MPI_rank == 0) {
		vector<int> final_result_vec;
		vector<int> final_result_count_vec;

		// merge results for each query
		for (int i = 0; i < config.num_of_queries; ++i) {
			// count is first, id is second (sort by count)
			priority_queue<pii, vector<pii>, std::less<pii> > single_query_priority_queue;
			for (int j = 0; j < MPI_size; ++j) {
				int offset = j * single_rank_result_size + i * config.num_of_topk;
				for (int k = 0; k < config.num_of_topk; ++k) {
					// k-th result on j-th rank for i-th query
					single_query_priority_queue.push(pii(final_result_count[offset + k], final_result[offset + k]));
				}
			}

			// append top k of i-th query to final result vector (which contains top k of all queries)
			for (int j = 0; j < config.num_of_topk; ++j) {
				pii single_result = single_query_priority_queue.top();
				single_query_priority_queue.pop();
				final_result_vec.push_back(single_result.second);
				final_result_count_vec.push_back(single_result.first);
			}
		}

		// debug
		//for (auto it = final_result_vec.begin(); it < final_result_vec.end(); ++it)
		//	cout << MPI_DEBUG << "final result vector: " << *it << endl;
		//for (auto it = final_result_count_vec.begin(); it < final_result_count_vec.end(); ++it)
		//	cout << MPI_DEBUG << "final result count vector: " << *it << endl;

		// write result to file
		ofstream output(extra_config.output_file);
		for (auto it1 = final_result_vec.begin(), it2 = final_result_count_vec.begin(); it1 != final_result_vec.end(); ++it1, ++it2) {
			output << *it1 << "," << *it2 << endl;
		}
		output.close();
	}

	delete[] table;
	MPI_Finalize();
	return EXIT_SUCCESS;
}

/*
 * Parse configuration file
 *
 * param config (OUTPUT) Config struct of GPUGenie
 * param extra_config (OUTPUT) Extra configuration for MPIGenie
 * param config_filename (INPUT) Configuration file name
 */
void ParseConfigurationFile(
		GPUGenie_Config &config,
		ExtraConfig &extra_config,
		const string config_filename)
{
	/*
	 * read configurations from file and store them in a map
	 */
	map<string, string> config_map;
	ifstream config_file(config_filename);
	string line, key, value;
	while (getline(config_file, line))
	{
		istringstream line_string_stream(line);
		if (getline(line_string_stream, key, '='))
			if (getline(line_string_stream, value))
				config_map[key] = value;
	}
	config_file.close();

	if (!ValidateConfiguration(config_map))
	{
		MPI_Finalize();
		return;
	}

	/*
	 * set configuration structs accordingly
	 */
	extra_config.data_file = config_map.find("data_file")->second;
	extra_config.query_file = config_map.find("query_file")->second;
	extra_config.output_file = config_map.find("output_file")->second;

	config.dim = stoi(config_map.find("dim")->second);
	config.count_threshold = stoi(config_map.find("count_threshold")->second);
	config.num_of_topk = stoi(config_map.find("num_of_topk")->second);
	config.hashtable_size = 14 * config.num_of_topk * 1.5;
	config.query_radius = 0;
	config.use_device = LOCAL_RANK;
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;
	
	config.use_load_balance = false;
	config.posting_list_max_length = 6400;
	config.multiplier = 1.5f;
	config.use_multirange = false;
	
	config.data_type = stoi(config_map.find("data_type")->second);
	config.search_type = stoi(config_map.find("search_type")->second);
	config.max_data_size = stoi(config_map.find("max_data_size")->second);
	
	config.num_of_queries = stoi(config_map.find("num_of_queries")->second);
}

/*
 * Checks whether all compulsory entries are present
 *
 * param config_map (INPUT) map of configurations
 */
bool ValidateConfiguration(map<string, string> config_map)
{
	vector<string> compulsoryEntries;
	compulsoryEntries.push_back("data_file");
	compulsoryEntries.push_back("query_file");
	compulsoryEntries.push_back("dim");
	compulsoryEntries.push_back("count_threshold");
	compulsoryEntries.push_back("num_of_topk");
	compulsoryEntries.push_back("num_of_queries");
	compulsoryEntries.push_back("data_type");
	compulsoryEntries.push_back("search_type");
	compulsoryEntries.push_back("max_data_size");
	compulsoryEntries.push_back("output_file");

	for (auto iterator = compulsoryEntries.begin(); iterator < compulsoryEntries.end(); ++iterator)
	{
		if (config_map.find(*iterator) == config_map.end())
		{
			cout << MPI_DEBUG << "Configuration validation failed" << endl;
			return false;
		}
	}
	cout << MPI_DEBUG << "Configuration validation succeeded" << endl;
	return true;
}
