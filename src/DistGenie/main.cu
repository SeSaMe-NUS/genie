#include <mpi.h>
#include <iostream>
#include <cstdio>
#include "GPUGenie.h"

#define LOCAL_RANK atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"))

using namespace std;
using namespace GPUGenie;

void ParseConfigurationFile(GPUGenie_Config &, const string);

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
	config.query_points = &queries;
	config.data_points = &data;
	string configFileName(argv[1]);
	ParseConfigurationFile(config, configFileName);
	cout << "rank: " << MPI_rank << " using GPU " << config.use_device << endl;

	/*
	 * load data
	 */
	string data_file = "static/sift_20_" + to_string(MPI_rank) + ".csv"; // each process load a different file
	string query_file = "static/sift_20.csv";
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
void ParseConfigurationFile(GPUGenie_Config &config, const string configFileName)
{
	// TODO: parse the configuration file
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
}
