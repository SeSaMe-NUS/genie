#ifndef __DISTGENIE_CONTAINER_H__
#define __DISTGENIE_CONTAINER_H__

#include <vector>

namespace DistGenie
{
	struct Cluster {
		std::vector<int> m_queries_id{};
		std::vector<std::vector<int> > m_queries{};
	};

	typedef std::vector<std::pair<int, int> > Result;
}

#endif
