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
	return *this;
}

Config& genie::Config::SetQueryRange(uint32_t query_range)
{
	query_range_ = query_range;
	return *this;
}

Config& genie::Config::SetNumOfQuery(uint32_t num_of_query)
{
	num_of_query_ = num_of_query;
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
