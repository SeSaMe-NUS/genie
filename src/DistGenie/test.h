#ifndef __DISTGENIE_TEST_H__
#define __DISTGENIE_TEST_H__

#include <vector>
#include "container.h"

void MergeResult(vector<distgenie::Result> &results, vector<vector<int> > &h_topk, vector<vector<int> > &h_topk_count,
		int topk, vector<distgenie::Cluster> &clusters, vector<int> &id_offset);

#endif
