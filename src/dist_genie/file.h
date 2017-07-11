#ifndef __DISTGENIE_FILE_H__
#define __DISTGENIE_FILE_H__

#include <vector>
#include <genie/GPUGenie.h>
#include "config.h"
#include "container.h"

namespace distgenie
{
	namespace file
	{
		void ReadData(genie::original::GPUGenie_Config &, distgenie::DistGenieConfig &,
                std::vector<std::vector<int> > &, std::vector<std::shared_ptr<genie::table::inv_table>> &);
		void GenerateOutput(std::vector<distgenie::Result> &, genie::original::GPUGenie_Config &,
                distgenie::DistGenieConfig &);
	}
}

#endif
