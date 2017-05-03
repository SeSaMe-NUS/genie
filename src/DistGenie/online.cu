#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <array>
#include <thread>
#include <queue>
#include <mutex>
#include <chrono> // benchmarking purpose

#include "GPUGenie.h"
#include "distgenie.h"
#define NO_EXTERN
#include "global.h"

using namespace GPUGenie;
using namespace distgenie;
using namespace std;

static const size_t BUFFER_SIZE = 10u << 20;

static void ParseQueryAndSearch(int *, array<char,BUFFER_SIZE> &, GPUGenie_Config &, ExtraConfig &, vector<inv_table*> &, vector<Cluster> &, vector<int> &);

static void WaitForGDB()
{
	if(getenv("ENABLE_GDB") != NULL && 0 == g_mpi_rank){
		volatile int gdb_attached = 0;
		fprintf(stderr, "Process %ld waiting for GDB \n", (long)getpid());
		fflush(stderr);
		
		while (0 == gdb_attached)
			sleep(5);

		fprintf(stderr, "Process %ld got GDB \n", (long)getpid());
		fflush(stderr);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

static void CalculateIdOffset(vector<int> &id_offset, vector<inv_table*> &tables)
{
	/* mpi operations */
	int *local_tablesize = new int[tables.size()];
	for (size_t i = 0; i < tables.size(); ++i)
		local_tablesize[i] = tables.at(i)->i_size();
	int *global_tablesize = new int[tables.size() * g_mpi_size];
	MPI_Allgather(local_tablesize, tables.size(), MPI_INT, global_tablesize, tables.size(), MPI_INT, MPI_COMM_WORLD);

	/* prefix sum */
	int sum = 0;
	for (size_t i = 0; i != tables.size() * g_mpi_size; ++i)
	{
		id_offset[i] = sum;
		sum += global_tablesize[i];
	}

	delete[] local_tablesize;
	delete[] global_tablesize;
}

int main(int argc, char *argv[])
{
	/*
	 * initialization
	 */
	MPI_Init(&argc, &argv);
	WaitForGDB();

	MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
	if (argc != 2)
	{
		if (0 == g_mpi_rank)
			cout << "Usage: mpirun -np <proc> --hostfile <hosts> ./dgenie config.file" << endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/*
	 * process configuration file
	 */
	GPUGenie_Config config;
	ExtraConfig extra_config;
	string config_filename(argv[1]);
	parser::ParseConfigurationFile(config, extra_config, config_filename);

	/*
	 * initialize GENIE
	 */
	init_genie(config);
	vector<vector<int> > data;
	config.data_points = &data;
	vector<inv_table*> tables(extra_config.num_of_cluster);
	file::ReadData(config, extra_config, data, tables);
	vector<int> id_offset(extra_config.num_of_cluster * g_mpi_size);
	CalculateIdOffset(id_offset, tables);

	/*
	 * handle online queries
	 */
	int count;
	auto *recv_buf_ptr = new array<char,BUFFER_SIZE>();
	array<char,BUFFER_SIZE> &recv_buf = *recv_buf_ptr;
	vector<Cluster> clusters(extra_config.num_of_cluster);

	if (0 == g_mpi_rank)
	{
		queue<string> query_queue;
		thread scheduler(scheduler::ListenForQueries, ref(query_queue));

		while (true) {
			while (true) {
				query_mutex.lock();
				if (query_queue.empty())
				{
					query_mutex.unlock();
					this_thread::sleep_for(chrono::seconds(1));
				}
				else
				{
					string &query_str = query_queue.front();
					count = query_str.length() + 1;
					memcpy(recv_buf.data(), query_str.c_str(), count);
					query_queue.pop();
					query_mutex.unlock();
					break;
				}
			}

			// set up output file name
			MPI_Barrier(MPI_COMM_WORLD);
			auto epoch_time = chrono::system_clock::now().time_since_epoch();
			auto output_filename = chrono::duration_cast<chrono::seconds>(epoch_time).count();
			extra_config.output_file = to_string(output_filename) + ".csv";

			ParseQueryAndSearch(&count, recv_buf, config, extra_config, tables, clusters, id_offset);
		}
	}
	else
	{
		while (true)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			ParseQueryAndSearch(&count, recv_buf, config, extra_config, tables, clusters, id_offset);
		}
	}
}

static void ParseQueryAndSearch(int *count_ptr, array<char,BUFFER_SIZE> &recv_buf, GPUGenie_Config &config,
		ExtraConfig &extra_config, vector<inv_table*> &tables, vector<Cluster> &clusters, vector<int> &id_offset)
{
	// broadcast query
	MPI_Bcast(count_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (g_mpi_rank != 0)
		memset(recv_buf.data(), '\0', BUFFER_SIZE);
	MPI_Bcast(recv_buf.data(), *count_ptr, MPI_CHAR, 0, MPI_COMM_WORLD);

	if(!parser::ValidateAndParseQuery(config, extra_config, clusters, string(recv_buf.data())))
		return;

	vector<Result> results(extra_config.total_queries);
	auto t1 = chrono::steady_clock::now();
	search::ExecuteMultitableQuery(config, extra_config, tables, clusters, results, id_offset);
	if (0 == g_mpi_rank)
	{
		auto t2 = chrono::steady_clock::now();
		auto diff = t2 - t1;
		clog << "Elapsed time: " << chrono::duration_cast<chrono::milliseconds>(diff).count() << "ms" << endl;
		file::GenerateOutput(results, config, extra_config);
		clog << "Output generated" << endl;
	}
}
