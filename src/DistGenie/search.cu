#include <algorithm>
#include <mpi.h>
#include <vector>

#include "search.h"
#include "sorting.h"
#include "global.h"

using namespace GPUGenie;
using namespace std;

/*
 * Load queries for different tables.
 */
static void LoadQueries(GPUGenie_Config &config, vector<shared_ptr<inv_table>> &tables,
	vector<distgenie::Cluster> &clusters, vector<vector<query> > &queries)
{
	for (vector<distgenie::Cluster>::size_type i = 0; i < clusters.size(); ++i)
	{
		config.num_of_queries = clusters.at(i).m_queries.size();
		config.query_points = &clusters.at(i).m_queries;
		queries.at(i).clear();
		load_query(*(tables.at(i)), queries.at(i), config);
	}
}


/*
 * Execute the query search on multiple tables. It
 * builds the queries and calls GENIE to search them.
 */
void distgenie::search::ExecuteMultitableQuery(GPUGenie_Config &config, DistGenieConfig &extra_config,
		vector<shared_ptr<GPUGenie::inv_table>> &tables, vector<Cluster> &clusters, vector<Result> &results,
		vector<int> &id_offset)
{
	vector<vector<query> > queries(clusters.size());
	LoadQueries(config, tables, clusters, queries);

	vector<vector<int> > h_topk(clusters.size());
	vector<vector<int> > h_topk_count(clusters.size());
	vector<GPUGenie_Config> configs;
	for (size_t i = 0; i < clusters.size(); ++i)
		configs.push_back(config);
	
	vector<inv_table*> raw_tables(tables.size());
	transform(tables.begin(), tables.end(), raw_tables.begin(), [](shared_ptr<inv_table> sp) {return sp.get();});

	knn_search_MT(raw_tables, queries, h_topk, h_topk_count, configs);
	MergeResult(results, h_topk, h_topk_count, config.num_of_topk, clusters, id_offset);
}
