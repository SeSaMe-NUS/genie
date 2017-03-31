#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include "rapidjson/document.h"

#include "parser.h"
#include "global.h"

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
void ParseConfigurationFile(GPUGenie_Config &config, ExtraConfig &extra_config, const string config_filename)
{
	/*
	 * read json configuration from file and parse it
	 */
	ifstream config_file(config_filename);
	string config_file_content((istreambuf_iterator<char>(config_file)), istreambuf_iterator<char>());
	config_file.close();
	Document json_config;
	if (json_config.Parse(config_file_content.c_str()).HasParseError())
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	/*
	 * validate the configuration
	 */
	if (!ValidateConfiguration(json_config))
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	cout << MPI_DEBUG << g_mpi_rank << " configuration validated" << endl;

	/*
	 * set configuration structs accordingly
	 */
	extra_config.data_file = json_config["data_file"].GetString();
	extra_config.num_of_cluster = json_config["num_of_cluster"].GetInt();

	config.dim = json_config["dim"].GetInt();
	config.count_threshold = json_config["count_threshold"].GetInt();
	config.query_radius = 0;
	config.use_device = LOCAL_RANK;
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
	/*
	 * maintain entries of different type in different vectors
	 */
	vector<string> string_entries, int_entries;
	string_entries.push_back("data_file");
	int_entries.push_back("dim");
	int_entries.push_back("count_threshold");
	int_entries.push_back("data_type");
	int_entries.push_back("search_type");
	int_entries.push_back("max_data_size");
	int_entries.push_back("num_of_cluster");

	/*
	 * check entries' existence and type
	 */
	for (auto it = string_entries.begin(); it != string_entries.end(); ++it) {
		if (!json_config.HasMember(it->c_str())) {
			cout << "Entry " << it->c_str() << " is missing" << endl;
			return false;
		}
		if (!json_config[it->c_str()].IsString()) {
			cout << "Entry " << it->c_str() << " has wrong type" << endl;
			return false;
		}	
	}
	for (auto it = int_entries.begin(); it != int_entries.end(); ++it) {
		if (!json_config.HasMember(it->c_str())) {
			cout << "Entry " << it->c_str() << " is missing" << endl;
			return false;
		}
		if (!json_config[it->c_str()].IsInt()) {
			cout << "Entry " << it->c_str() << " has wrong type" << endl;
			return false;
		}	
	}
	return true;
}

/*
 * Parse query into vector
 */
bool ValidateAndParseQuery(GPUGenie::GPUGenie_Config &config, vector<vector<int> > &queries, const string query)
{
	Document json_query;
	if (json_query.Parse(query.c_str()).HasParseError()) {
		cout << "Invalid query format" << endl;
		return false;
	}

	// validation
	if (!json_query.HasMember("topk")) {
		cout << "Entry topk is missing" << endl;
		return false;
	}
	if (!json_query["topk"].IsInt()) {
		cout << "Entry topk has wrong type" << endl;
		return false;
	}
	if (!json_query.HasMember("queries")) {
		cout << "Entry query is missing" << endl;
		return false;
	}
	if (!json_query["queries"].IsArray()) {
		cout << "Entry queries has wrong type" << endl;
		return false;
	}
	for (auto &single_query : json_query["queries"].GetArray()) {
		if (!single_query.IsArray()) {
			cout << "Entry queries has wrong type" << endl;
			return false;
		}
	}

	int topk, num_of_queries;
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

	return true;
}

} // end of namespace DistGenie
