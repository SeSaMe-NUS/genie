#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <genie/interface/genie.h>

using namespace std;
using namespace genie;

int main(int argc, char* argv[])
{
	namespace po = boost::program_options;

	uint32_t k, query_range, num_of_queries, gpu_id;
	string table_file, query_file;

	// this is for both CLI & file
	po::options_description generic;
	generic.add_options()
		("k,k", po::value<uint32_t>(&k), "k")
		("query-range,r", po::value<uint32_t>(&query_range), "query range")
		("num-of-queries,n", po::value<uint32_t>(&num_of_queries), "number of queries")
		("gpu", po::value<uint32_t>(&gpu_id), "GPU to use")
		("table,t", po::value<string>(&table_file), "table file")
		("query,q", po::value<string>(&query_file), "query file")
	;

	// generic + extra options for CLI only
	po::options_description cmdline_options("Allowed options");
	cmdline_options.add(generic).add_options()
		("help", "produce help message")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
	ifstream ifs("genie.cfg");
	po::store(po::parse_config_file(ifs, generic), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << cmdline_options << endl;
		return EXIT_SUCCESS;
	}

	if (!vm.count("table") || !vm.count("query"))
	{
		cout << "Table file or query file not specified." << endl;
		return EXIT_FAILURE;
	}

	Config config = Config();
	if (vm.count("k"))
		config.SetK(k);
	if (vm.count("num-of-queries"))
		config.SetNumOfQueries(num_of_queries);
	if (vm.count("query_range"))
		config.SetQueryRange(query_range);
	if (vm.count("gpu"))
		config.SetGpuId(gpu_id);

	// use try catch to display the errors nicely
	try
	{
		shared_ptr<ExecutionPolicy> policy = MakePolicy(config);
		config.DisplayConfiguration();
		SearchResult result = Search(policy, table_file, query_file);
	}
	catch (exception &e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
