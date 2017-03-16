#ifndef DistGenie_search_h
#define DistGenie_search_h

#include "GPUGenie.h"
#include "config.h"

namespace DistGenie
{

void ExecuteQuery(GPUGenie::GPUGenie_Config &, ExtraConfig &, GPUGenie::inv_table *);

}

#endif
