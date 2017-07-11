#include <memory>
#include <iostream>
#include <genie/genie.h>

using namespace genie;
using namespace std;

int main()
{
	// 1st step:
	//
	// set up GENIE with a genie::Config object
	//
	// generate a policy containing the correct implementation for
	// building table, building queries, and matching according to
	// the configurations specified
	Config config = Config()
		.SetK(5)
		.SetNumOfQueries(5);
	shared_ptr<ExecutionPolicy> policy = MakePolicy(config);

	// 2nd step:
	//
	// use the 2nd-level interface to adapt genie into your workflow.
	// e.g. you could use it to run different batches of queries on
	// the same table. or load a previously serialized table from binary
	// instead of building a new one from the CSV
	//
	// here we first load the raw table & query data
	//
	// then build table and queries, and match them
	TableData table_data = LoadTableDataFromCsv("../static/sift_20.csv");
	QueryData query_data = LoadQueryDataFromCsv("../static/sift_20.csv", policy);

	shared_ptr<genie::table::inv_table> table = BuildTable(policy, table_data);
	vector<genie::query::Query> queries = BuildQuery(policy, table, query_data);
	SearchResult results = Match(policy, table, queries);
	

	// 3rd step:
	//
	// the result is a pair of vectors. use .first to access the vector
	// of ids and .second to access the vector of counts
	cout << "First match has id: " << results.first[0] << " and count: " << results.second[0] << endl;

	return EXIT_SUCCESS;
}
