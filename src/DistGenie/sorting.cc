#include <mpi.h>
#include <vector>
#if defined(__GNUC__) && !defined(__NVCC__)
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include <functional>

#include "sorting.h"
#include "container.h"
#include "global.h"

using namespace std;
using namespace distgenie;

/*
 * Merge result from multiple nodes and tables
 */
void MergeResult(vector<Result> &results, vector<vector<int> > &h_topk, vector<vector<int> > &h_topk_count,
		int topk, vector<Cluster> &clusters, vector<int> &id_offset)
{
	/* build custom MPI datatype for std::pair<int, int> */
	MPI_Datatype mpi_pair;
	MPI_Datatype array_of_types[2] = {MPI_INT, MPI_INT};
	int array_of_blocklengths[2] = {1, 1};
	MPI_Aint array_of_displacements[2] = {0, sizeof(int)};
	MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &mpi_pair);
	MPI_Type_commit(&mpi_pair);

	//vector<int> local_results_topk, local_results_topk_count;
	//local_results_topk.reserve(results.size() * topk);
	//local_results_topk_count.reserve(results.size() * topk);
	vector<Result> local_results(results.size());

	/* merge result for a query from multiple clusters */
	for (size_t c = 0; c < clusters.size(); ++c)
	{
		int num_of_queries = clusters.at(c).m_queries.size();
		for (int i = 0; i < num_of_queries; ++i)
			for (int k = 0; k < topk; ++k)
				local_results.at(clusters.at(c).m_queries_id.at(i)).emplace_back(
						std::make_pair(h_topk_count.at(c).at(i * topk + k),
							id_offset.at(c * g_mpi_size + g_mpi_rank) + h_topk.at(c).at(i * topk + k)));
	}

	/* sort local result and gather on master */
	size_t i = 0;
	MPI_Request gather_request;
	MPI_Status gather_status;
	for (auto &&single_result : local_results)
	{
#if defined(__GNUC__) && !defined(__NVCC__)
		__gnu_parallel::sort(single_result.begin(), single_result.end(), std::greater<std::pair<int, int> >());
#else
		sort(single_result.begin(), single_result.end(), std::greater<std::pair<int, int> >());
#endif

		if (0 == g_mpi_rank)
			results.at(i).resize(topk * g_mpi_size);
		MPI_Igather(single_result.data(), topk, mpi_pair, results.at(i).data(), topk, mpi_pair, 0, MPI_COMM_WORLD, &gather_request);
		++i;
	}
	MPI_Wait(&gather_request, &gather_status); /* wait for the last MPI_Igather */

	//for (auto &&single_result : local_results)
	//	for (int j = 0; j < topk; ++j)
	//	{
	//		local_results_topk.push_back(single_result.at(j).second);
	//		local_results_topk_count.push_back(single_result.at(j).first);
	//	}

	///* gather local results */
	//vector<int> final_result, final_result_count;
	//if (0 == g_mpi_rank)
	//{
	//	final_result.resize(results.size() * topk * g_mpi_size);
	//	final_result_count.resize(results.size() * topk * g_mpi_size);
	//}
	//MPI_Gather(local_results_topk.data(), results.size() * topk, MPI_INT, final_result.data(), results.size() * topk, MPI_INT, 0, MPI_COMM_WORLD);
	//MPI_Gather(local_results_topk_count.data(), results.size() * topk, MPI_INT, final_result_count.data(), results.size() * topk, MPI_INT, 0, MPI_COMM_WORLD);

	/* put in the global result vector */
	//if (0 == g_mpi_rank)
	//{
	//	for (size_t i = 0; i < results.size(); ++i)
	//	{
	//		results.at(i).reserve(topk * g_mpi_size);
	//		for (int j = 0; j < g_mpi_size; ++j)
	//			for (int k = 0; k < topk; ++k)
	//				results.at(i).push_back(std::make_pair(
	//							final_result_count.at(results.size() * topk * j + topk * i + k),
	//							final_result.at(results.size() * topk * j + topk * i + k)
	//				));
	//	}
	//}
}
