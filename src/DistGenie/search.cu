#include <mpi.h>
#include <omp.h>
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
void ExecuteQuery(GPUGenie_Config &config, ExtraConfig &extra_config, inv_table *table, vector<Result> &results, vector<int> &query_id)
{
	/*
	 * Search step
	 */
	vector<int> result;
	vector<int> result_count;
	MPI_Barrier(MPI_COMM_WORLD);
	knn_search_after_preprocess(config, table, result, result_count);

	/*
	 * Collect results from all GPUs
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
		// add result from a single cluster to results vector
		for (int i = 0; i < config.num_of_queries; ++i)
			for (int j = 0; j < g_mpi_size; ++j) {
				int offset = j * single_rank_result_size + i * config.num_of_topk;
				for (int k = 0; k < config.num_of_topk; ++k)
					results.at(query_id.at(i)).push_back(pii(final_result_count[offset + k], final_result[offset + k]));
			}

		// cleanup
		delete[] final_result;
		delete[] final_result_count;
	}
}

/* For pre-clustering version */
void ExecuteMultitableQuery(GPUGenie::GPUGenie_Config &config, ExtraConfig &extra_config,
		GPUGenie::inv_table **tables, vector<Cluster> &clusters, vector<Result> &results)
{
	vector<vector<query> > queries(clusters.size());
#pragma omp parallel for schedule(dynamic)
	for (vector<Cluster>::size_type i = 0; i < clusters.size(); ++i)
	{
		clog << MPI_DEBUG << g_mpi_rank << " searching cluster " << i << endl;
		config.num_of_queries = clusters.at(i).m_queries.size();
		config.query_points = &clusters.at(i).m_queries;
		//ExecuteQuery(config, extra_config, tables[i], results, clusters.at(i).m_queries_id);
		queries.at(i).clear();
		load_query(tables[i][0], queries.at(i), config);
	}
}

} // end of namespace DistGenie
