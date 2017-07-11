#ifndef __DISTGENIE_SORTING_H__
#define __DISTGENIE_SORTING_H__

#include <vector>
#include "container.h"

void MergeResult(std::vector<distgenie::Result> &results, std::vector<std::vector<int> > &h_topk,
		std::vector<std::vector<int> > &h_topk_count, int topk, std::vector<distgenie::Cluster> &clusters,
		std::vector<int> &id_offset);

#endif
