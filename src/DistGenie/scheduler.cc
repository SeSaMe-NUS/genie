#include <iostream>
#include <cstring>
#include <string>
#include <mutex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <mpi.h>

#include "scheduler.h"
#include "global.h"

using namespace std;

void distgenie::scheduler::ListenForQueries(queue<string> &query_queue)
{
	/* initialize socket */
	auto *socket_buf_ptr = new array<char,BUFFER_SIZE>();
	auto &socket_buf = *socket_buf_ptr;
	int sock = socket(PF_INET, SOCK_STREAM, 0);
	sockaddr_in address;
	sockaddr client_address;
	socklen_t address_len = sizeof(client_address);

	address.sin_family = AF_INET;
	address.sin_port = htons(9090);
	address.sin_addr.s_addr = INADDR_ANY;
	int status;
	status = bind(sock, (struct sockaddr *)&address, sizeof(address));
	if (-1 == status)
	{
		clog << "Socket bind() failed" << endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	status = listen(sock, 1);
	if (-1 == status)
	{
		clog << "Socket listen() failed" << endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	char hostname[128];
	gethostname(hostname, sizeof(hostname));

	while (true)
	{
		/* receive data */
		clog << "Accepting queries on " << hostname << ":9090" << endl;
		int incoming = accept(sock, &client_address, &address_len);
		memset(socket_buf.data(), '\0', BUFFER_SIZE);
		recv(incoming, socket_buf.data(), BUFFER_SIZE, MSG_WAITALL);
		close(incoming);
		clog << "Received query, scheduled to run" << endl;

		/* put data to query queue */
		lock_guard<mutex> lock(query_mutex);
		query_queue.emplace(string(socket_buf.data()));
	}
}
