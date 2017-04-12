#ifndef __DISTGENIE_SEARCH_H__
#define __DISTGENIE_SEARCH_H__

#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace DistGenie
{
	//void ExecuteQuery(GPUGenie::GPUGenie_Config &, ExtraConfig &, GPUGenie::inv_table *, vector<Result> &);
	void ExecuteMultitableQuery(GPUGenie::GPUGenie_Config &, ExtraConfig &, std::vector<GPUGenie::inv_table*> &, vector<Cluster> &, vector<Result> &);
}

#endif
