#ifndef __DISTGENIE_FILE_H__
#define __DISTGENIE_FILE_H__

#include <vector>
#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace distgenie
{
	namespace file
	{
		void ReadData(GPUGenie::GPUGenie_Config &, distgenie::DistGenieConfig &, std::vector<std::vector<int> > &, std::vector<std::shared_ptr<GPUGenie::inv_table>> &);
		void GenerateOutput(std::vector<distgenie::Result> &, GPUGenie::GPUGenie_Config &, distgenie::DistGenieConfig &);
	}
}

#endif
