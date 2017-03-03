#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>
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
	cout << "Reading configurations from " << config_filename << endl;
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
			return false;
	}
	cout << "Configuration validated" << endl;
	return true;
}
