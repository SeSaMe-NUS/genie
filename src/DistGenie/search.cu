#include <mpi.h>
#include <queue>
#include <vector>
#include <fstream>
#include <functional>

#include "search.h"
#include "global.h"

#define pii std::pair<int, int>

using namespace GPUGenie;
using namespace std;

namespace DistGenie
{
/*
 * Execute queries
 * param config Config struct for GPUGenie
 * param extra_config Extra config struct used by MPI program
 * param table The inverted list to search
 */
void ExecuteQuery(GPUGenie_Config &config, ExtraConfig &extra_config, inv_table *table)
{
	/*
	 * Search step
	 */
	vector<int> result;
	vector<int> result_count;
	MPI_Barrier(MPI_COMM_WORLD);
	knn_search_after_preprocess(config, table, result, result_count);

	/*
	 * merge results from all ranks to get global top k
	 */
	int *final_result = nullptr;       // only for MPI
	int *final_result_count = nullptr; // only for MPI
	int single_rank_result_size = config.num_of_topk * config.num_of_queries;
	if (g_mpi_rank == 0) {
		int result_size = g_mpi_size * single_rank_result_size;
		final_result = new int[result_size];
		final_result_count = new int[result_size];
	}
	MPI_Gather(&result[0], single_rank_result_size, MPI_INT, final_result, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&result_count[0], single_rank_result_size, MPI_INT, final_result_count, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);

	if (g_mpi_rank == 0) {
		vector<int> final_result_vec;
		vector<int> final_result_count_vec;

		// merge results for each query
		for (int i = 0; i < config.num_of_queries; ++i) {
			// count is first, id is second (sort by count)
			priority_queue<pii, vector<pii>, std::less<pii> > single_query_priority_queue;
			for (int j = 0; j < g_mpi_size; ++j) {
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

		// write result to file
		ofstream output(extra_config.output_file);
		for (auto it1 = final_result_vec.begin(), it2 = final_result_count_vec.begin(); it1 != final_result_vec.end(); ++it1, ++it2) {
			output << *it1 << "," << *it2 << endl;
		}
		output.close();

		// cleanup
		delete[] final_result;
		delete[] final_result_count;
	}
}

}
