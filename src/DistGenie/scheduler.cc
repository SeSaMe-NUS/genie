#include <iostream>
#include <cstring>
#include <string>
#include <mutex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include "scheduler.h"
#include "global.h"

using namespace std;

namespace distgenie
{
	void scheduler::ListenForQueries(queue<string> &query_queue)
	{
		const size_t BUFFER_SIZE = 10u << 20;
		/* initialize socket */
		auto *socket_buf_ptr = new array<char,BUFFER_SIZE>();
		array<char,BUFFER_SIZE> &socket_buf = *socket_buf_ptr;
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

		while (true)
		{
			/* receive data */
			clog << "Accepting queries on localhost:9090" << endl;
			int incoming = accept(sock, &client_address, &address_len);
			memset(socket_buf.data(), '\0', BUFFER_SIZE);
			recv(incoming, socket_buf.data(), BUFFER_SIZE, MSG_WAITALL);
			close(incoming);
			clog << "Received query, start processing" << endl;

			/* put data to query queue */
			lock_guard<mutex> lock(query_mutex);
			query_queue.emplace(string(socket_buf.data()));
		}
	}
} // end of namespace distgenie::scheduler
