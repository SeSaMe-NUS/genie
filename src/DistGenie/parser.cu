#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>

#include "parser.h"

#define LOCAL_RANK atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"))

using namespace GPUGenie;

namespace DistGenie
{
/*
 * Parse configuration file
 *
 * param config (OUTPUT) Config struct of GPUGenie
 * param extra_config (OUTPUT) Extra configuration for MPIGenie
 * param config_filename (INPUT) Configuration file name
 */
void ParseConfigurationFile(
		GPUGenie_Config &config,
		ExtraConfig &extra_config,
		const string config_filename)
{
	/*
	 * read configurations from file and store them in a map
	 */
	map<string, string> config_map;
	ifstream config_file(config_filename);
	string line, key, value;
	while (getline(config_file, line))
	{
		istringstream line_string_stream(line);
		if (getline(line_string_stream, key, '='))
			if (getline(line_string_stream, value))
				config_map[key] = value;
	}
	config_file.close();

	if (!ValidateConfiguration(config_map))
	{
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return;
	}

	/*
	 * set configuration structs accordingly
	 */
	extra_config.data_file = config_map.find("data_file")->second;

	config.dim = stoi(config_map.find("dim")->second);
	config.count_threshold = stoi(config_map.find("count_threshold")->second);
	config.query_radius = 0;
	config.use_device = LOCAL_RANK;
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;
	
	config.use_load_balance = false;
	config.posting_list_max_length = 6400;
	config.multiplier = 1.5f;
	config.use_multirange = false;
	//config.save_to_gpu = true;
	
	config.data_type = stoi(config_map.find("data_type")->second);
	config.search_type = stoi(config_map.find("search_type")->second);
	config.max_data_size = stoi(config_map.find("max_data_size")->second);
}

/*
 * Checks whether all compulsory entries are present
 *
 * param config_map (INPUT) map of configurations
 */
bool ValidateConfiguration(map<string, string> config_map)
{
	vector<string> compulsoryEntries;
	compulsoryEntries.push_back("data_file");
	compulsoryEntries.push_back("dim");
	compulsoryEntries.push_back("count_threshold");
	compulsoryEntries.push_back("data_type");
	compulsoryEntries.push_back("search_type");
	compulsoryEntries.push_back("max_data_size");

	for (auto iterator = compulsoryEntries.begin(); iterator < compulsoryEntries.end(); ++iterator)
	{
		if (config_map.find(*iterator) == config_map.end())
		{
			cout << "Configuration validation failed" << endl;
			return false;
		}
	}
	cout << "Configuration validation succeeded" << endl;
	return true;
}

}
