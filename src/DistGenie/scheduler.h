#ifndef __DISTGENIE_SCHEDULER_H__
#define __DISTGENIE_SCHEDULER_H__

#include <queue>
#include <string>

namespace distgenie
{
	void ListenForQueries(std::queue<std::string> &);
}

#endif
