#include <mpi.h>
#include <omp.h>
#include <queue>
#include <vector>
#include <fstream>
#include <functional>

#include "search.h"
#include "global.h"

using namespace DistGenie;
using namespace GPUGenie;
using namespace std;

/*
 * Execute queries
 * param config Config struct for GPUGenie
 * param extra_config Extra config struct used by MPI program
 * param table The inverted list to search
 */
//static void ExecuteQuery(GPUGenie_Config &config, ExtraConfig &extra_config, inv_table *table, vector<Result> &results, vector<int> &query_id)
//{
//	/*
//	 * Search step
//	 */
//	vector<int> result;
//	vector<int> result_count;
//	MPI_Barrier(MPI_COMM_WORLD);
//	knn_search_after_preprocess(config, table, result, result_count);
//
//	/*
//	 * Collect results from all GPUs
//	 */
//	int *final_result = nullptr;       // only for MPI
//	int *final_result_count = nullptr; // only for MPI
//	int single_rank_result_size = config.num_of_topk * config.num_of_queries;
//	if (g_mpi_rank == 0) {
//		int result_size = g_mpi_size * single_rank_result_size;
//		final_result = new int[result_size];
//		final_result_count = new int[result_size];
//	}
//	MPI_Gather(&result[0], single_rank_result_size, MPI_INT, final_result, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);
//	MPI_Gather(&result_count[0], single_rank_result_size, MPI_INT, final_result_count, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);
//
//	if (g_mpi_rank == 0) {
//		// add result from a single cluster to results vector
//		for (int i = 0; i < config.num_of_queries; ++i)
//			for (int j = 0; j < g_mpi_size; ++j) {
//				int offset = j * single_rank_result_size + i * config.num_of_topk;
//				for (int k = 0; k < config.num_of_topk; ++k)
//					results.at(query_id.at(i)).push_back(std::pair<int, int>(final_result_count[offset + k], final_result[offset + k]));
//			}
//
//		// cleanup
//		delete[] final_result;
//		delete[] final_result_count;
//	}
//}

/* load queries for different tables in parallel */
static void LoadQueries(GPUGenie_Config &config, vector<inv_table*> &tables, vector<Cluster> &clusters, vector<vector<query> > &queries)
{
//#pragma omp parallel for schedule(dynamic)
	for (vector<Cluster>::size_type i = 0; i < clusters.size(); ++i)
	{
		config.num_of_queries = clusters.at(i).m_queries.size();
		config.query_points = &clusters.at(i).m_queries;
		queries.at(i).clear();
		load_query(tables.at(i)[0], queries.at(i), config);
	}
}

/* Merge result for multi-node & multi-cluster */
void MergeResult(vector<Result> &results, vector<vector<int> > &h_topk, vector<vector<int> > &h_topk_count,
		int topk, vector<Cluster> &clusters, vector<int> &id_offset)
{
	int *final_result = nullptr;       // only for MPI
	int *final_result_count = nullptr; // only for MPI

	/* for each cluster */
	for (size_t c = 0; c < clusters.size(); ++c)
	{
		int num_of_queries = h_topk.at(c).size() / topk;
		int single_rank_result_size = h_topk.at(c).size();
		if (g_mpi_rank == 0) {
			int result_size = g_mpi_size * single_rank_result_size;
			final_result = new int[result_size];
			final_result_count = new int[result_size];
		}
		MPI_Gather(&h_topk.at(c)[0], single_rank_result_size, MPI_INT, final_result, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gather(&h_topk_count.at(c)[0], single_rank_result_size, MPI_INT, final_result_count, single_rank_result_size, MPI_INT, 0, MPI_COMM_WORLD);

		if (g_mpi_rank == 0) {
			// add result from a single cluster to results vector
			for (int i = 0; i < num_of_queries; ++i)
				for (int j = 0; j < g_mpi_size; ++j) {
					int offset = j * single_rank_result_size + i * topk;
					for (int k = 0; k < topk; ++k)
						results.at(clusters.at(c).m_queries_id.at(i)).push_back(
							std::pair<int, int>(final_result_count[offset + k],
								id_offset.at(j * clusters.size() + c) + final_result[offset + k]
							)
						);
				}

			// cleanup
			delete[] final_result;
			delete[] final_result_count;
		}
	}
}

/* For pre-clustering version */
void DistGenie::ExecuteMultitableQuery(GPUGenie::GPUGenie_Config &config, ExtraConfig &extra_config,
		vector<GPUGenie::inv_table*> &tables, vector<Cluster> &clusters, vector<Result> &results, vector<int> &id_offset)
{
	vector<vector<query> > queries(clusters.size());
	LoadQueries(config, tables, clusters, queries);

	vector<vector<int> > h_topk(clusters.size());
	vector<vector<int> > h_topk_count(clusters.size());
	vector<GPUGenie_Config> configs;
	for (size_t i = 0; i < clusters.size(); ++i)
		configs.push_back(config);

	knn_search_MT(tables, queries, h_topk, h_topk_count, configs);
	MergeResult(results, h_topk, h_topk_count, config.num_of_topk, clusters, id_offset);

	//for (auto it1 = queries.begin(); it1 != queries.end(); ++it1)
	//	for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
	//		for (size_t i = 0; i < it2->count_ranges(); i++)
	//			it2->clear_dim(i);
}
