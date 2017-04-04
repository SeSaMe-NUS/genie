#ifndef __DISTGENIE_PARSER_H__
#define __DISTGENIE_PARSER_H__

#include <vector>

#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace DistGenie
{
	void ParseConfigurationFile(GPUGenie::GPUGenie_Config &, ExtraConfig &, const string);
	bool ValidateAndParseQuery(GPUGenie::GPUGenie_Config &, ExtraConfig &, vector<Cluster> &, const string);
}

#endif
