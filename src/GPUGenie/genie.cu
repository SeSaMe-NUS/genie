#include <memory>
#include <string>
#include <iostream>

#include <GPUGenie/interface/genie.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace genie;

int main(int argc, char* argv[])
{
	namespace po = boost::program_options;

	uint32_t k, query_range, num_of_query, gpu_id;
	string table_file, query_file;

	po::options_description descriptions("Allowed options");
	descriptions.add_options()
		("help", "produce help message")
		("k,k", po::value<uint32_t>(&k), "k")
		("query-range,r", po::value<uint32_t>(&query_range), "query range")
		("num-of-query,n", po::value<uint32_t>(&num_of_query), "number of query")
		("gpu", po::value<uint32_t>(&gpu_id), "GPU to use")
		("table,t", po::value<string>(&table_file), "table file")
		("query,q", po::value<string>(&query_file), "query file")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, descriptions), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << descriptions << endl;
		return EXIT_SUCCESS;
	}

	if (!vm.count("table") || !vm.count("query"))
	{
		cout << "Table file or query file not specified" << endl;
		return EXIT_FAILURE;
	}

	// configure GENIE and get the execution policy
	Config config = Config();
	if (vm.count("k"))
		config.SetK(k);
	if (vm.count("num-of-query"))
		config.SetNumOfQuery(num_of_query);
	if (vm.count("query_range"))
		config.SetQueryRange(query_range);
	if (vm.count("gpu"))
		config.SetGpuId(gpu_id);
	shared_ptr<ExecutionPolicy> policy = MakePolicy(config);
	config.DisplayConfiguration();

	// search with GENIE using the execution policy
	SearchResult result = Search(policy, table_file, query_file);

	return EXIT_SUCCESS;
}
