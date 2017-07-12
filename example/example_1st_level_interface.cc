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
	// call genie::Search() to complete building table & query and
	// matching in one function. this is suitable for the simplest
	// use case of genie
	SearchResult results = Search(policy, "../static/sift_20.csv", "../static/sift_20.csv");

	// 3rd step:
	//
	// the result is a pair of vectors. use .first to access the vector
	// of ids and .second to access the vector of counts
	cout << "First match has id: " << results.first[0] << " and count: " << results.second[0] << endl;

	return EXIT_SUCCESS;
}
