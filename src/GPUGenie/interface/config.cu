#include "config.h"
#include <GPUGenie.h>
#include <GPUGenie/exception/exception.h>

using namespace genie;
using namespace std;

uint32_t genie::Config::GetK() const
{
	return k_;
}

uint32_t genie::Config::GetQueryRange() const
{
	return query_range_;
}

uint32_t genie::Config::GetNumOfQuery() const
{
	return num_of_query_;
}

bool genie::Config::GetSaveToGpu() const
{
	return save_to_gpu_;
}

uint8_t genie::Config::GetGpuId() const
{
	return gpu_id_;
}

Config& genie::Config::SetK(const uint32_t k)
{
	k_ = k;
	k_initialized_ = true;
	return *this;
}

Config& genie::Config::SetQueryRange(const uint32_t query_range)
{
	query_range_ = query_range;
	query_range_initialized_ = true;
	return *this;
}

Config& genie::Config::SetNumOfQuery(const uint32_t num_of_query)
{
	num_of_query_ = num_of_query;
	num_of_query_initialized_ = true;
	return *this;
}

Config& genie::Config::SetSaveToGpu(const bool save_to_gpu)
{
	save_to_gpu_ = save_to_gpu;
	return *this;
}

Config& genie::Config::SetGpuId(const uint8_t gpu_id)
{
	gpu_id_ = gpu_id;
	return *this;
}

Config& genie::Config::LoadFromFile(const string& filename)
{
	throw genie::exception::NotImplementedException();
}

bool genie::Config::IsKSet() const
{
	return k_initialized_;
}

bool genie::Config::IsQueryRangeSet() const
{
	return query_range_initialized_;
}

bool genie::Config::IsNumOfQuerySet() const
{
	return num_of_query_initialized_;
}

void genie::Config::Validate() const
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
