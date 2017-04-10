#ifndef __DISTGENIE_CONFIG_H__
#define __DISTGENIE_CONFIG_H__

namespace DistGenie
{
	struct ExtraConfig
	{
		string data_file;
		string query_file;
		string output_file;
		int num_of_cluster;
		int total_queries;
		int data_format;
	};
}

#endif
