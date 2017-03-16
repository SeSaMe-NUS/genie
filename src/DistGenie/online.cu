#include <mpi.h>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include "GPUGenie.h"
#include "DistGenie.h"

#define MPI_DEBUG ">>>MPI DEBUG<<< "

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
	 * MPI initialization
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
	cout << MPI_DEBUG << "rank: " << MPI_rank << " using GPU " << config.use_device << endl;

	/*
	 * load data
	 */
	string data_file = extra_config.data_file + "_" + to_string(MPI_rank) + ".csv"; // each process load a different file
	read_file(data, data_file.c_str(), -1);

	/*
	 * prepare the inverted list
	 */
	inv_table * table = nullptr;
	preprocess_for_knn_csv(config, table);

	/*
	 * handle online queries
	 */
	int num_of_queries, top_k;
	int * queries_array = nullptr;

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
		char * recv_buf = new char[1000];
		bind(sock, (struct sockaddr *)&address, sizeof(address));
		int status = listen(sock, 1);

		while (true) {
			/*
			 * receive queries
			 */
			int incoming = accept(sock, &client_address, &address_len);
			memset(recv_buf, 0, 1000);
			int count = recv(incoming, recv_buf, 1000, 0);
			close(incoming);

			// parse the request (format: num topk xx xx xx xx xx)
			string msg(recv_buf);
			istringstream msg_sstream(msg);
			int q_num;

			// num of queries
			msg_sstream >> num_of_queries;
			//cout << MPI_DEBUG << "num of queries: " << num_of_queries << endl;
			MPI_Bcast(&num_of_queries, 1, MPI_INT, 0, MPI_COMM_WORLD); // number of queries
			config.num_of_queries = num_of_queries;
			if (num_of_queries == -1) {
				close(sock);
				delete[] recv_buf;
				break;
			}

			// top k
			msg_sstream >> top_k;
			//cout << MPI_DEBUG << "topk: " << top_k << endl;
			MPI_Bcast(&top_k, 1, MPI_INT, 0, MPI_COMM_WORLD); // top k
			config.num_of_topk = top_k;

			// query content
			try {
				queries_array = new int[num_of_queries * config.dim];
			} catch (bad_alloc&) {
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
			//queries_array = (int *)realloc(queries_array, sizeof(int) * num_of_queries * config.dim);
			for (int i = 0; i < num_of_queries; ++i)
				for (int j = 0; j < config.dim; ++j) {
					msg_sstream >> q_num;
					queries_array[i * config.dim + j] = q_num;	
				}
			MPI_Bcast(queries_array, config.num_of_queries * config.dim, MPI_INT, 0, MPI_COMM_WORLD); // actual queries as 1d array

			// set up config values
			config.hashtable_size = config.num_of_topk * 1.5 * config.count_threshold;
			extra_config.output_file = "GENIEQUERY.csv"; // TODO: change filename according to current time?

			// convert queries_array to vector
			queries.clear();
			for (int i = 0; i < config.num_of_queries; ++i) {
				vector<int> single_query(queries_array + i * config.dim, queries_array + (i + 1) * config.dim);
				queries.push_back(single_query);
			}
			delete[] queries_array;
			// run the queries and write output to file
			ExecuteQuery(config, extra_config, table);
		}
	} else {
		while (true) {
			/*
			 * receive query from master
			 */
			// num of queries
			MPI_Bcast(&num_of_queries, 1, MPI_INT, 0, MPI_COMM_WORLD); // number of queries
			config.num_of_queries = num_of_queries;
			//cout << MPI_DEBUG << MPI_rank << " num of queries: " << config.num_of_queries << endl;
			if (num_of_queries == -1)
				break;

			// top k
			MPI_Bcast(&top_k, 1, MPI_INT, 0, MPI_COMM_WORLD); // top k
			config.num_of_topk = top_k;
			//cout << MPI_DEBUG << MPI_rank << " topk: " << config.num_of_topk << endl;

			// query content
			try {
				queries_array = new int[config.num_of_queries * config.dim];
			} catch (bad_alloc&) {
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
			//queries_array = (int *)realloc(queries_array, sizeof(int) * num_of_queries + config.dim);
			//cout << MPI_DEBUG << MPI_rank << " allocated query space " << config.num_of_queries * config.dim << endl;
			MPI_Bcast(queries_array, config.num_of_queries * config.dim, MPI_INT, 0, MPI_COMM_WORLD); // actual queries as 1d array
			//cout << MPI_DEBUG << MPI_rank << " after query broadcast" << endl;
			//for (int i = 0; i < config.num_of_queries * config.dim; ++i)
			//	cout << MPI_DEBUG << MPI_rank << " received: " << queries_array[i] << endl;

			// set up config values
			config.hashtable_size = config.num_of_topk * 1.5 * config.count_threshold;
			//cout << MPI_DEBUG << MPI_rank << " after hash size update" << endl;

			// convert queries_array to vector
			queries.clear();
			for (int i = 0; i < config.num_of_queries; ++i) {
				vector<int> single_query(queries_array + i * config.dim, queries_array + (i + 1) * config.dim);
				//cout << MPI_DEBUG << MPI_rank << " before pushing one query" << endl;
				//queries.push_back(single_query);
				queries.push_back(single_query);
				//cout << MPI_DEBUG << MPI_rank << " pushed one query" << endl;
			}
			//cout << MPI_DEBUG << MPI_rank << " after vector conversion" << endl;
			delete[] queries_array;
			// run the queries and write output to file
			//cout << MPI_DEBUG << MPI_rank << " before executing query" << endl;
			ExecuteQuery(config, extra_config, table);
			//cout << MPI_DEBUG << MPI_rank << " after executing query" << endl;
		}
	}

	/*
	 * clean up
	 */
	delete[] table;
	MPI_Finalize();
	return EXIT_SUCCESS;
}
