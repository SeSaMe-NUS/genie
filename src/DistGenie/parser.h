#ifndef __DISTGENIE_PARSER_H__
#define __DISTGENIE_PARSER_H__

#include <vector>

#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace distgenie
{
	namespace parser
	{
		void ParseConfigurationFile(GPUGenie::GPUGenie_Config &, DistGenieConfig &, const string);
		bool ValidateAndParseQuery(GPUGenie::GPUGenie_Config &, DistGenieConfig &, vector<Cluster> &, const string);
	}
}

#endif
