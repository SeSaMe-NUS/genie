#include <mpi.h>
#include <vector>
#include <algorithm>
//#include <parallel/algorithm>
#include <functional>

#include "search.h"
#include "sorting.h"
#include "global.h"

using namespace GPUGenie;
using namespace std;

/* load queries for different tables in parallel */
static void LoadQueries(GPUGenie_Config &config, vector<inv_table*> &tables, vector<distgenie::Cluster> &clusters, vector<vector<query> > &queries)
{
//#pragma omp parallel for schedule(dynamic)
	for (vector<distgenie::Cluster>::size_type i = 0; i < clusters.size(); ++i)
	{
		config.num_of_queries = clusters.at(i).m_queries.size();
		config.query_points = &clusters.at(i).m_queries;
		queries.at(i).clear();
		load_query(tables.at(i)[0], queries.at(i), config);
	}
}


/* For pre-clustering version */
void distgenie::search::ExecuteMultitableQuery(GPUGenie_Config &config, ExtraConfig &extra_config,
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
}
