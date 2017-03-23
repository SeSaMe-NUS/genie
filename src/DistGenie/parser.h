#ifndef DistGenie_parser_h
#define DistGenie_parser_h

#include <vector>

#include "GPUGenie.h"
#include "config.h"

namespace DistGenie
{

void ParseConfigurationFile(GPUGenie::GPUGenie_Config &, ExtraConfig &, const string);
bool ValidateAndParseQuery(GPUGenie::GPUGenie_Config &, vector<vector<int> > &, const string);

}

#endif
