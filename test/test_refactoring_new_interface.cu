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
		.SetQueryRange(0)
		.SetNumOfQuery(5)
		.SetGpuId(0);
	shared_ptr<ExecutionPolicy> policy = ExecutionPolicyFactory::MakePolicy(config);

	// search with GENIE using the execution policy
	string table_file = "../static/sift_20.csv";
	string query_file = "../static/sift_20.csv";
	SearchResult result = Search(policy, table_file, query_file);

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
