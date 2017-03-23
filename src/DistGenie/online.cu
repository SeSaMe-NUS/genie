#include <mpi.h>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <chrono> // benchmarking purpose

#include "GPUGenie.h"
#include "DistGenie.h"

#define MPI_DEBUG ">>>MPI DEBUG<<< "
#define BUFFER_SIZE 2 << 20

using namespace GPUGenie;
using namespace DistGenie;
using namespace std;

static void WaitForGDB()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(getenv("ENABLE_GDB") != NULL && rank == 0){
		volatile int gdb_attached =0;
		fprintf(stderr, "Process %ld waiting for GDB \n", (long)getpid());
		fflush(stderr);
		
		while (gdb_attached==0)
			sleep(5);

		fprintf(stderr, "Process %ld got GDB \n", (long)getpid());
		fflush(stderr);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char *argv[])
{
	/*
	 * initialization
	 */
	int MPI_rank, MPI_size;
	MPI_Init(&argc, &argv);
	WaitForGDB();

	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
	if (argc != 2)
	{
		if (MPI_rank == 0)
			cout << "Usage: mpirun -np <proc> --hostfile <hosts> ./dgenie config.file" << endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/*
	 * read in the config
	 */
	vector< vector<int> > queries;
	vector< vector<int> > data;
	GPUGenie_Config config;
	ExtraConfig extra_config;
	string config_filename(argv[1]);

	config.query_points = &queries;
	config.data_points = &data;
	ParseConfigurationFile(config, extra_config, config_filename);

	/*
	 * load data
	 */
	string data_file = extra_config.data_file + "_" + to_string(MPI_rank) + ".csv"; // each process load a different file
	read_file(data, data_file.c_str(), -1);

	/*
	 * prepare the inverted list
	 */
	inv_table *table = nullptr;
	init_genie(config);
	preprocess_for_knn_csv(config, table);

	/*
	 * handle online queries
	 */
	int count;
	char *queries_array = nullptr;

	if (MPI_rank == 0) {
		// socket
		// TODO: check socket success
		int sock = socket(PF_INET, SOCK_STREAM, 0);
		sockaddr_in address;
		sockaddr client_address;
		socklen_t address_len = sizeof(client_address);

		address.sin_family = AF_INET;
		address.sin_port = htons(9090);
		address.sin_addr.s_addr = INADDR_ANY;
		char *recv_buf = new char[BUFFER_SIZE];
		bind(sock, (struct sockaddr *)&address, sizeof(address));
		int status = listen(sock, 1);

		while (true) {
			//receive queries from socket
			int incoming = accept(sock, &client_address, &address_len);
			memset(recv_buf, 0, BUFFER_SIZE);
			count = recv(incoming, recv_buf, BUFFER_SIZE, 0);
			close(incoming);

			// broadcast query length and query content
			MPI_Bcast(&count, 1, MPI_INT, 0 , MPI_COMM_WORLD);
			MPI_Bcast(recv_buf, count, MPI_CHAR, 0, MPI_COMM_WORLD);

			// parse the query
			if (!ValidateAndParseQuery(config, queries, string(recv_buf)))
				continue;

			// set up config values
			extra_config.output_file = "GENIEQUERY.csv"; // TODO: change filename according to current time?

			// run the queries and write output to file
			for (int i = 0; i < 10; ++i) {
				auto start = chrono::steady_clock::now();
				ExecuteQuery(config, extra_config, table);
				auto stop = chrono::steady_clock::now();
				auto diff = stop - start;
				cout << "Elapsed time is " << chrono::duration_cast<chrono::milliseconds>(diff).count() << "ms" << endl;
			}
		}
	} else {
		while (true) {
			// receive query from master
			MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
			queries_array = new char[count];
			MPI_Bcast(queries_array, count, MPI_CHAR, 0, MPI_COMM_WORLD);

			// parse the query
			bool validated = ValidateAndParseQuery(config, queries, string(queries_array));
			delete[] queries_array;
			if (!validated)
				continue;

			// run the queries and write output to file
			for (int i = 0; i < 10; ++i)
				ExecuteQuery(config, extra_config, table);
		}
	}
}
