#ifndef __DISTGENIE_CONFIG_H__
#define __DISTGENIE_CONFIG_H__

#include <string>

namespace distgenie
{
	struct ExtraConfig
	{
		std::string data_file;
		std::string query_file;
		std::string output_file;
		int num_of_cluster;
		int total_queries;
		int data_format;
	};
}

#endif
