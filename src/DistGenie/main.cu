#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "GPUGenie.h"

#define LOCAL_RANK atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
#define MPI_DEBUG ">>>MPI DEBUG<<< "

using namespace std;
using namespace GPUGenie;

struct ExtraConfig {
	string data_file;
	string query_file;
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
	inv_table * table = NULL;
	vector<int> result;
	vector<int> result_count;
	preprocess_for_knn_csv(config, table);
	knn_search_after_preprocess(config, table, result, result_count);

	// debug
	//for (auto it = result_count.begin(); it < result_count.end(); ++it)
	//	cout << MPI_DEBUG << "result from rank " << MPI_rank << ": " << *(it) << endl;
	//cout << MPI_DEBUG << "result from rank " << MPI_rank << ": " << result_count.size() << endl;

	/*
	 * TODO: merge results from all ranks
	 */
	int *final_result;
	int *final_result_count;
	int single_rank_result_size = config.num_of_topk * config.num_of_queries;
	int result_size;
	if (MPI_rank == 0) {
		result_size = MPI_size * single_rank_result_size;
		final_result = (int *)malloc(sizeof(int) * result_size);
		final_result_count = (int *)malloc(sizeof(int) * result_size);
	}
	MPI_Gather(&result[0], single_rank_result_size, MPI_INT, final_result, result_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&result_count[0], single_rank_result_size, MPI_INT, final_result_count, result_size, MPI_INT, 0, MPI_COMM_WORLD);

	// sort after gather
	if (MPI_rank == 0) {
		vector<int> single_query_result;
		vector<int> single_query_result_count;
		vector<int> final_result_vec;
		vector<int> final_result_count_vec;

		// sort for each query and concatenate them
		for (int i = 0; i < config.num_of_queries; ++i) {
			single_query_result.clear();
			for (int j = 0; j < MPI_size; ++j) {
				for (int k = 0; k < config.num_of_topk; ++k) {
					int offset = j * single_rank_result_size + i * config.num_of_topk + k; // k-th result on j-th rank for i-th query
					single_query_result.push_back(final_result[offset]);
				}
			}
			sort(single_query_result.begin(), single_query_result.end(), std::greater<int>());
			single_query_result.resize(config.num_of_topk); // only save the overall top-k results
			cout << MPI_DEBUG << "single query of " << i << " has size " << single_query_result.size() << endl;
			for (int x = 0; x < config.num_of_topk; ++x)
				cout << MPI_DEBUG << "query[" << x << "]: " << single_query_result.at(x) << endl;
			final_result_vec.insert(final_result_vec.end(), single_query_result.begin(), single_query_result.end()); // final vector contains top-k for all queries, so append to it
		}

		for (auto it = final_result_vec.begin(); it < final_result_vec.end(); ++it)
			cout << MPI_DEBUG << "final result vec: " << *it << endl;
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}

/*
 * Parse configuration file
 *
 * param config (OUTPUT) Config struct of GPUGenie
 * param configFileName (INPUT) Configuration file name
 */
void ParseConfigurationFile(
		GPUGenie_Config &config,
		ExtraConfig &extra_config,
		const string config_filename)
{
	/*
	 * read configurations from file and store them in a map
	 */
	cout << MPI_DEBUG << "Reading configurations from " << config_filename << endl;
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

	/*
	 * create config structs from configuration map
	 */
	if (!ValidateConfiguration(config_map))
		MPI_Finalize();
	config.dim = 5;
	config.count_threshold = 14;
	config.num_of_topk = 5;
	config.hashtable_size = 14*config.num_of_topk*1.5;
	config.query_radius = 0;
	config.use_device = LOCAL_RANK;
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;
	
	config.use_load_balance = false;
	config.posting_list_max_length = 6400;
	config.multiplier = 1.5f;
	config.use_multirange = false;
	
	config.data_type = 0;
	config.search_type = 0;
	config.max_data_size = 0;
	
	config.num_of_queries = 3;

	extra_config.data_file = config_map.find("data_file")->second;
	extra_config.query_file = config_map.find("query_file")->second;
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
	compulsoryEntries.push_back("num_of_queries");

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
