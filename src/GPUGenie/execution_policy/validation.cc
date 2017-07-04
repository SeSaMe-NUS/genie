#include "validation.h"
#include <GPUGenie/exception/exception.h>

using namespace genie;

void genie::execution_policy::validation::ValidateK(uint32_t k)
{
	if (k <= 0)
		throw exception::InvalidConfigurationException("K should be larger than 0.");
}

void genie::execution_policy::validation::ValidateNumOfQuery(uint32_t num_of_query)
{
	if (num_of_query <= 0)
		throw exception::InvalidConfigurationException("Number of query should be larger than 0.");
}

void genie::execution_policy::validation::ValidateQueryRange(uint32_t query_range)
{
	if (query_range <= 0)
		throw exception::InvalidConfigurationException("Query range should be larger than 0.");
}
