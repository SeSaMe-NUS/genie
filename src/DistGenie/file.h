#ifndef __DISTGENIE_FILE_H__
#define __DISTGENIE_FILE_H__

#include <vector>
#include "GPUGenie.h"
#include "config.h"
#include "container.h"

namespace distgenie
{
	void ReadData(GPUGenie::GPUGenie_Config &, distgenie::ExtraConfig &, std::vector<std::vector<int> > &, std::vector<GPUGenie::inv_table*> &);
	void GenerateOutput(std::vector<distgenie::Result> &, GPUGenie::GPUGenie_Config &, distgenie::ExtraConfig &);
}

#endif
