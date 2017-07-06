#ifndef __DISTGENIE_SEARCH_H__
#define __DISTGENIE_SEARCH_H__

#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace distgenie
{
	namespace search
	{
		void ExecuteMultitableQuery(GPUGenie::GPUGenie_Config &, DistGenieConfig &,
            std::vector<std::shared_ptr<GPUGenie::inv_table>> &, vector<Cluster> &, vector<Result> &, vector<int> &);
	}
}

#endif
