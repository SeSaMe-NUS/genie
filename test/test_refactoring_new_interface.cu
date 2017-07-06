// test 20 - refactoring
//
// This is to test the new GENIE interface with
// execution policies.
//
// This test searches for exactly the same thing
// as test 1.

#undef NDEBUG

#include <memory>
#include <iostream>
#include <cassert>
#include <GPUGenie/interface/genie.h>

using namespace std;
using namespace genie;

int main()
{
	// configure GENIE and get the execution policy
	Config config = Config()
		.SetK(5)
		.SetNumOfQuery(5);
	shared_ptr<ExecutionPolicy> policy = MakePolicy(config);
	config.DisplayConfiguration();

	// search with GENIE using the execution policy
	SearchResult result = Search(policy, "../static/sift_20.csv", "../static/sift_20.csv");

	assert(result.first[0] == 0);
	assert(result.second[0] == 5);
	assert(result.first[1] == 4);
	assert(result.second[1] == 2);
	assert(result.first[5] == 1);
	assert(result.second[5] == 5);
	assert(result.first[10] == 2);
	assert(result.second[10] == 5);

	return EXIT_SUCCESS;
}
