#ifndef __DISTGENIE_SEARCH_H__
#define __DISTGENIE_SEARCH_H__

#include <genie/GPUGenie.h>
#include "config.h"
#include "container.h"

namespace distgenie
{
	namespace search
	{
		void ExecuteMultitableQuery(genie::original::GPUGenie_Config &, DistGenieConfig &,
            std::vector<std::shared_ptr<genie::table::inv_table>> &, std::vector<Cluster> &, std::vector<Result> &,
            std::vector<int> &);
	}
}

#endif
