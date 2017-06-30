#include "config.h"
#include <GPUGenie/exception/exception.h>

using namespace genie;
using namespace std;

uint32_t genie::Config::GetK()
{
	return k_;
}

uint32_t genie::Config::GetQueryRange()
{
	return query_range_;
}

uint32_t genie::Config::GetNumOfQuery()
{
	return num_of_query_;
}

bool genie::Config::GetSaveToGpu()
{
	return save_to_gpu_;
}

uint8_t genie::Config::GetGpuId()
{
	return gpu_id_;
}

Config& genie::Config::SetK(uint32_t k)
{
	k_ = k;
	k_initialized_ = true;
	return *this;
}

Config& genie::Config::SetQueryRange(uint32_t query_range)
{
	query_range_ = query_range;
	query_range_initialized_ = true;
	return *this;
}

Config& genie::Config::SetNumOfQuery(uint32_t num_of_query)
{
	num_of_query_ = num_of_query;
	num_of_query_initialized_ = true;
	return *this;
}

Config& genie::Config::SetSaveToGpu(bool save_to_gpu)
{
	save_to_gpu_ = save_to_gpu;
	return *this;
}

Config& genie::Config::SetGpuId(uint8_t gpu_id)
{
	gpu_id_ = gpu_id;
	return *this;
}

Config& genie::Config::LoadFromFile(string& filename)
{
	throw genie::exception::NotImplementedException();
}

bool genie::Config::IsKSet()
{
	return k_initialized_;
}

bool genie::Config::IsQueryRangeSet()
{
	return query_range_initialized_;
}

bool genie::Config::IsNumOfQuerySet()
{
	return num_of_query_initialized_;
}

void genie::Config::Validate()
{
	if (!k_initialized_)
		throw exception::InvalidConfigurationException("K is not set.");
	if (!num_of_query_initialized_)
		throw exception::InvalidConfigurationException("Number of query is not set.");
}

void genie::Config::DisplayConfiguration()
{
	Logger::log(Logger::INFO, "GENIE configuration:");
	Logger::log(Logger::INFO, "K: %d", k_);
	Logger::log(Logger::INFO, "Number of query: %d", num_of_query_);
	if (IsQueryRangeSet())
		Logger::log(Logger::INFO, "Query range: %d", query_range_);
	Logger::log(Logger::INFO, "Save to GPU: %s", save_to_gpu_ ? "true" : "false");
	Logger::log(Logger::INFO, "GPU used: %d", gpu_id_);
	Logger::log(Logger::INFO, "");
}
