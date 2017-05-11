#include <mpi.h>
#include <vector>
#include <parallel/algorithm>
#include <functional>

#include "container.h"
#include "global.h"

using namespace std;
using namespace distgenie;

/* Merge result for multi-node & multi-cluster */
void MergeResult(vector<distgenie::Result> &results, vector<vector<int> > &h_topk, vector<vector<int> > &h_topk_count,
		int topk, vector<distgenie::Cluster> &clusters, vector<int> &id_offset)
{
	int *final_result = nullptr;
	int *final_result_count = nullptr;
	int *local_results_topk = new int[results.size() * topk];
	int *local_results_topk_count = new int[results.size() * topk];
	vector<distgenie::Result> local_results(results.size());
	//vector<thrust::device_vector<thrust::pair<int,int> > > local_results_d(results.size());

	/* for each cluster */
	for (size_t c = 0; c < clusters.size(); ++c)
	{
		/* put data into local result vector */
		int num_of_queries = clusters.at(c).m_queries.size();
		for (size_t i = 0; i < num_of_queries; ++i)
			for (size_t k = 0; k < topk; ++k)
				local_results.at(clusters.at(c).m_queries_id.at(i)).push_back(
						std::make_pair(h_topk_count.at(c).at(i * topk + k),
							id_offset.at(c * g_mpi_size + g_mpi_rank) + h_topk.at(c).at(i * topk + k)
						)
				);
	}

	//for (size_t i = 0; i < local_results.size(); ++i)
	//{
	//	local_results_d.at(i).resize(local_results.at(i).size());
	//	thrust::copy(local_results.at(i).begin(), local_results.at(i).end(), local_results_d.at(i).begin());
	//	thrust::sort(local_results_d.at(i).begin(), local_results_d.at(i).end(), thrust::greater<thrust::pair<int, int> >());
	//	//thrust::copy(local_results_d.at(i).begin(), local_results_d.at(i).end(), local_results.at(i).begin());
	//}
	for (auto it = local_results.begin(); it != local_results.end(); ++it)
		std::__parallel::sort(it->begin(), it->end(), std::greater<std::pair<int, int> >());
		//std::sort(it->begin(), it->end(), std::greater<std::pair<int, int> >());

	for (size_t i = 0; i < local_results.size(); ++i)
		for (size_t j = 0; j < topk; ++j)
		{
			local_results_topk[i * topk + j] = local_results.at(i).at(j).second;
			local_results_topk_count[i * topk + j] = local_results.at(i).at(j).first;
		}

	/* gather local results */
	if (0 == g_mpi_rank)
	{
		final_result = new int[results.size() * topk * g_mpi_size];
		final_result_count = new int[results.size() * topk * g_mpi_size];
	}
	MPI_Gather(local_results_topk, results.size() * topk, MPI_INT, final_result, results.size() * topk, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(local_results_topk_count, results.size() * topk, MPI_INT, final_result_count, results.size() * topk, MPI_INT, 0, MPI_COMM_WORLD);

	/* put in the global result vector */
	if (0 == g_mpi_rank)
	{
		for (size_t i = 0; i < results.size(); ++i)
		{
			results.at(i).reserve(topk * g_mpi_size);
			for (size_t j = 0; j < g_mpi_size; ++j)
				for (size_t k = 0; k < topk; ++k)
					results.at(i).push_back(std::make_pair(
								final_result_count[results.size() * topk * j + topk * i + k],
								final_result[results.size() * topk * j + topk * i + k]
					));
		}
		delete[] final_result;
		delete[] final_result_count;
	}

	delete[] local_results_topk;
	delete[] local_results_topk_count;
}
