#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>

#include "rapidjson/document.h"
#include "parser.h"

#define LOCAL_RANK atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"))

using namespace GPUGenie;
using namespace rapidjson;
using namespace std;

namespace DistGenie
{
bool ValidateConfiguration(const Document &);
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
	 * read json configuration and parse it
	 */
	ifstream config_file(config_filename);
	string config_file_content((istreambuf_iterator<char>(config_file)), istreambuf_iterator<char>());
	Document json_config;
	json_config.Parse(config_file_content.c_str());
	config_file.close();

	/*
	 * validate the configuration
	 */
	if (!ValidateConfiguration(json_config))
	{
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return;
	}
	cout << "Configuration validated" << endl;

	/*
	 * set configuration structs accordingly
	 */
	extra_config.data_file = json_config["data_file"].GetString();

	config.dim = json_config["dim"].GetInt();
	config.count_threshold = json_config["count_threshold"].GetInt();
	config.query_radius = 0;
	config.use_device = LOCAL_RANK + 1; // TODO: change it back to LOCAL_RANK
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;
	
	config.use_load_balance = false;
	config.posting_list_max_length = 6400;
	config.multiplier = 1.5f;
	config.use_multirange = false;
	config.save_to_gpu = true;
	
	config.data_type = json_config["data_type"].GetInt();
	config.search_type = json_config["search_type"].GetInt();
	config.max_data_size = json_config["max_data_size"].GetInt();
}

/*
 * Checks whether all compulsory entries are present
 *
 * param json_config (INPUT) JSON config document
 */
bool ValidateConfiguration(const Document &json_config)
{
	// TODO: validate data type
	vector<string> compulsoryEntries;
	compulsoryEntries.push_back("data_file");
	compulsoryEntries.push_back("dim");
	compulsoryEntries.push_back("count_threshold");
	compulsoryEntries.push_back("data_type");
	compulsoryEntries.push_back("search_type");
	compulsoryEntries.push_back("max_data_size");

	for (auto iterator = compulsoryEntries.begin(); iterator < compulsoryEntries.end(); ++iterator)
		if (!json_config.HasMember((*iterator).c_str()))
			return false;
	return true;
}

/*
 * Parse query into vector
 */
void ParseQuery(GPUGenie::GPUGenie_Config &config, vector<vector<int> > &queries, const string query)
{
	// TODO: add validation
	int topk, num_of_queries;

	Document json_query;
	json_query.Parse(query.c_str());

	topk = json_query["topk"].GetInt();
	num_of_queries = json_query["queries"].Size();

	queries.clear();
	for (auto &single_query : json_query["queries"].GetArray()) {
		vector<int> single_query_vector;
		for (auto &query_value : single_query.GetArray()) 
			single_query_vector.push_back(query_value.GetInt());
		queries.push_back(single_query_vector);
	}

	config.num_of_queries = num_of_queries;
	config.num_of_topk = topk;
	config.hashtable_size = config.num_of_topk * 1.5 * config.count_threshold;
}

} // end of namespace DistGenie
