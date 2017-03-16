#ifndef DistGenie_parser_h
#define DistGenie_parser_h

#include <map>
#include "GPUGenie.h"
#include "config.h"

namespace DistGenie
{

void ParseConfigurationFile(GPUGenie::GPUGenie_Config &, ExtraConfig &, const string);
bool ValidateConfiguration(map<string, string>);

}

#endif
