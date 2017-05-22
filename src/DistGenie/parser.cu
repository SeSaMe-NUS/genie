#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include "rapidjson/document.h"

#include "parser.h"
#include "global.h"

const int LOCAL_RANK = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

using namespace GPUGenie;
using namespace rapidjson;
using namespace std;

/*
 * Checks whether all compulsory entries
 * are present in configuration file
 */
static bool ValidateConfiguration(const Document &json_config)
{
	vector<string> string_entries, int_entries;
	string_entries.push_back("data_file");
	int_entries.push_back("dim");
	int_entries.push_back("count_threshold");
	int_entries.push_back("data_type");
	int_entries.push_back("search_type");
	int_entries.push_back("max_data_size");
	int_entries.push_back("num_of_cluster");
	int_entries.push_back("data_format");

	for (auto it = string_entries.begin(); it != string_entries.end(); ++it) {
		if (!json_config.HasMember(it->c_str())) {
			if (0 == g_mpi_rank)
				cout << "Entry " << it->c_str() << " is missing" << endl;
			return false;
		}
		if (!json_config[it->c_str()].IsString()) {
			if (0 == g_mpi_rank)
				cout << "Entry " << it->c_str() << " has wrong type" << endl;
			return false;
		}	
	}

	for (auto it = int_entries.begin(); it != int_entries.end(); ++it) {
		if (!json_config.HasMember(it->c_str())) {
			if (0 == g_mpi_rank)
				cout << "Entry " << it->c_str() << " is missing" << endl;
			return false;
		}
		if (!json_config[it->c_str()].IsInt()) {
			if (0 == g_mpi_rank)
				cout << "Entry " << it->c_str() << " has wrong type" << endl;
			return false;
		}	
	}

	return true;
}

/*
 * Parse configuration file
 */
void distgenie::parser::ParseConfigurationFile(GPUGenie_Config &config, DistGenieConfig &extra_config, const string config_filename)
{
	/* read json configuration from file and parse it */
	ifstream config_file(config_filename);
	string config_file_content((istreambuf_iterator<char>(config_file)), istreambuf_iterator<char>());
	config_file.close();
	Document json_config;
	if (json_config.Parse(config_file_content.c_str()).HasParseError())
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	/* validate the configuration */
	if (!ValidateConfiguration(json_config))
	{
		if (0 == g_mpi_rank)
			clog << "Configuration file validation failed" << endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/* set configuration structs accordingly */
	extra_config.data_file = json_config["data_file"].GetString();
	extra_config.num_of_cluster = json_config["num_of_cluster"].GetInt();
	extra_config.data_format = json_config["data_format"].GetInt();

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
 * Parse query into vector
 */
bool distgenie::parser::ValidateAndParseQuery(GPUGenie_Config &config, DistGenieConfig &extra_config, vector<Cluster> &clusters, const string query)
{
	Document json_query;
	if (json_query.Parse(query.c_str()).HasParseError()) {
		if (0 == g_mpi_rank)
			cout << "Received query is not a valid JSON document" << endl;
		return false;
	}

	/* validation */
	if (!json_query.HasMember("topk")) {
		if (0 == g_mpi_rank)
			cout << "Entry 'topk' is missing" << endl;
		return false;
	}
	if (!json_query["topk"].IsInt()) {
		if (0 == g_mpi_rank)
			cout << "Entry 'topk' should be an interger" << endl;
		return false;
	}
	if (!json_query.HasMember("queries")) {
		if (0 == g_mpi_rank)
			cout << "Entry 'queries' is missing" << endl;
		return false;
	}
	if (!json_query["queries"].IsArray()) {
		if (0 == g_mpi_rank)
			cout << "Entry 'queries' should be an array" << endl;
		return false;
	}
	else
	{
		for (auto &&single_query_json : json_query["queries"].GetArray())
		{
			if (!single_query_json.HasMember("content"))
			{
				if (0 == g_mpi_rank)
					cout << "Some query misses the 'content' section" << endl;
				return false;
			}
			if (!single_query_json["content"].IsArray())
			{
				if (0 == g_mpi_rank)
					cout << "Query's 'content' section should be an array" << endl;
				return false;
			}
			if (!single_query_json.HasMember("clusters"))
			{
				if (0 == g_mpi_rank)
					cout << "Some query misses the 'clusters' section" << endl;
				return false;
			}
			if (!single_query_json["clusters"].IsArray())
			{
				if (0 == g_mpi_rank)
					cout << "Query's 'clusters' section should be an array" << endl;
				return false;
			}
		}
	}

	int topk;
	topk = json_query["topk"].GetInt();
	extra_config.total_queries = json_query["queries"].Size();

	for (auto &&cluster : clusters)
	{
		cluster.m_queries.clear();
		cluster.m_queries_id.clear();
	}
	int id = 0;
	for (auto &&single_query_json : json_query["queries"].GetArray()) {
		vector<int> single_query_content;
		int cluster_id;
		for (auto &&query_value : single_query_json["content"].GetArray()) 
			single_query_content.emplace_back(query_value.GetInt());
		for (auto &&cluster : single_query_json["clusters"].GetArray()) 
		{
			cluster_id = cluster.GetInt();
			clusters.at(cluster_id).m_queries.emplace_back(single_query_content);
			clusters.at(cluster_id).m_queries_id.emplace_back(id);
		}
		++id;
	}

	config.num_of_topk = topk;
	config.hashtable_size = config.num_of_topk * 1.5 * config.count_threshold;

	return true;
}
