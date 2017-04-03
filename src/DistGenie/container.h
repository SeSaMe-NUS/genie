#ifndef __DISTGENIE_CONTAINER_H__
#define __DISTGENIE_CONTAINER_H__

#include <utility>

namespace DistGenie
{
	struct Cluster {
		vector<int> m_queries_id{};
		vector<vector<int> > m_queries{};
	};
	
	struct Result {
		vector<std::pair<int, int> > m_results{};
	};
}

#endif
