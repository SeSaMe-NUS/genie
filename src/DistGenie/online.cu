#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <chrono> // benchmarking purpose

#include "GPUGenie.h"
#include "DistGenie.h"
#define NO_EXTERN
#include "global.h"

using namespace GPUGenie;
using namespace DistGenie;
using namespace std;

static const size_t BUFFER_SIZE = 10u << 20; // 10 megabytes

static void ParseQueryAndSearch(int *, char *, GPUGenie_Config &, ExtraConfig &, vector<inv_table*> &, vector<Cluster> &, vector<int> &);

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
	ParseConfigurationFile(config, extra_config, config_filename);

	/*
	 * initialize GENIE
	 */
	init_genie(config);
	vector<vector<int> > data;
	config.data_points = &data;
	vector<inv_table*> tables(extra_config.num_of_cluster);
	ReadData(config, extra_config, data, tables);
	vector<int> id_offset(extra_config.num_of_cluster * g_mpi_size);
	CalculateIdOffset(id_offset, tables);

	/*
	 * handle online queries
	 */
	int count;
	char *recv_buf = new char[BUFFER_SIZE]{'\0'};
	vector<Cluster> clusters(extra_config.num_of_cluster);

	if (0 == g_mpi_rank)
	{
		// TODO: check socket success
		int sock = socket(PF_INET, SOCK_STREAM, 0);
		sockaddr_in address;
		sockaddr client_address;
		socklen_t address_len = sizeof(client_address);

		address.sin_family = AF_INET;
		address.sin_port = htons(9090);
		address.sin_addr.s_addr = INADDR_ANY;
		bind(sock, (struct sockaddr *)&address, sizeof(address));
		int status = listen(sock, 1);

		while (true) {
			//receive queries from socket
			MPI_Barrier(MPI_COMM_WORLD);
			clog << "Accepting queries on localhost:9090" << endl;
			int incoming = accept(sock, &client_address, &address_len);
			memset(recv_buf, '\0', BUFFER_SIZE);
			count = recv(incoming, recv_buf, BUFFER_SIZE, MSG_WAITALL);
			close(incoming);
			clog << "Received query, start processing" << endl;

			// set up output file name
			auto epoch_time = chrono::system_clock::now().time_since_epoch();
			auto output_filename = chrono::duration_cast<chrono::seconds>(epoch_time).count();
			extra_config.output_file = to_string(output_filename) + ".csv";

			ParseQueryAndSearch(&count, recv_buf, config, extra_config, tables, clusters, id_offset);
		}
	}
	else
	{
		MPI_Barrier(MPI_COMM_WORLD);
		while (true)
			ParseQueryAndSearch(&count, recv_buf, config, extra_config, tables, clusters, id_offset);
	}
}

static void ParseQueryAndSearch(int *count_ptr, char *recv_buf, GPUGenie_Config &config,
		ExtraConfig &extra_config, vector<inv_table*> &tables, vector<Cluster> &clusters, vector<int> &id_offset)
{
	// broadcast query
	MPI_Bcast(count_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (g_mpi_rank != 0)
		memset(recv_buf, '\0', BUFFER_SIZE);
	MPI_Bcast(recv_buf, *count_ptr, MPI_CHAR, 0, MPI_COMM_WORLD);

	if(!ValidateAndParseQuery(config, extra_config, clusters, string(recv_buf)))
		return;

	vector<Result> results(extra_config.total_queries);
	auto t1 = chrono::steady_clock::now();
	ExecuteMultitableQuery(config, extra_config, tables, clusters, results, id_offset);
	if (0 == g_mpi_rank)
	{
		auto t2 = chrono::steady_clock::now();
		auto diff = t2 - t1;
		clog << "Elapsed time: " << chrono::duration_cast<chrono::milliseconds>(diff).count() << "ms" << endl;
		GenerateOutput(results, config, extra_config);
		clog << "Output generated" << endl;
	}
}
