#ifndef __DISTGENIE_SCHEDULER_H__
#define __DISTGENIE_SCHEDULER_H__

#include <queue>
#include <string>

namespace distgenie
{
	namespace scheduler
	{
		void ListenForQueries(std::queue<std::string> &);
	}
}

#endif
