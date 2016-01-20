#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include "GPUGenie/lib/bucket_topk/bucket_topk.h" //for ide: to revert it as system file later, change  "GPUGenie/lib/bucket_topk/bucket_topk.h" to "lib/bucket_topk/bucket_topk.h"

#include "topk.h"

#include "Logger.h"
#include "Timing.h"

struct ValueOfFloat
{
	float max;__host__ __device__ float valueOf(float data)
	{
		return (float) max - data;
	}
};
struct ValueOfU32
{
	u32 max;__host__ __device__ u32 valueOf(u32 data)
	{
		return (u32) max - data;
	}
};

struct ValueOfData
{
	float max;__host__ __device__ float valueOf(data_t data)
	{
		return (float) max - data.aggregation;
	}
};

struct ValueOfInt
{
	float max;__host__ __device__ float valueOf(int data)
	{
		return (float) max - data;
	}
};

void GPUGenie::topk(device_vector<int>& d_search,
		vector<GPUGenie::query>& queries, device_vector<int>& d_top_indexes)
{
	host_vector<int> h_tops(queries.size());
	unsigned int i;
	for (i = 0; i < queries.size(); i++)
	{
		h_tops[i] = queries[i].topk();
	}
	device_vector<int> d_tops(h_tops);
	topk(d_search, d_tops, d_top_indexes);
}

void GPUGenie::topk(device_vector<float>& d_search,
		vector<GPUGenie::query>& queries, device_vector<int>& d_top_indexes)
{
	host_vector<int> h_tops(queries.size());
	unsigned int i;
	for (i = 0; i < queries.size(); i++)
	{
		h_tops[i] = queries[i].topk();
	}
	device_vector<int> d_tops(h_tops);
	topk(d_search, d_tops, d_top_indexes);
}

void GPUGenie::topk(device_vector<data_t>& d_search,
		vector<GPUGenie::query>& queries, device_vector<int>& d_top_indexes,
		float dim)
{
	host_vector<int> h_tops(queries.size());
	unsigned int i;
	for (i = 0; i < queries.size(); i++)
	{
		h_tops[i] = queries[i].topk();
	}
	device_vector<int> d_tops(h_tops);
	topk(d_search, d_tops, d_top_indexes, dim);
}

void GPUGenie::topk(device_vector<u32>& d_search,
		vector<GPUGenie::query>& queries, device_vector<int>& d_top_indexes,
		u32 dim)
{
	host_vector<int> h_tops(queries.size());
	unsigned int i;
	for (i = 0; i < queries.size(); i++)
	{
		h_tops[i] = queries[i].topk();
	}
	device_vector<int> d_tops(h_tops);
	topk(d_search, d_tops, d_top_indexes, dim);
}

void GPUGenie::topk(device_vector<int>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes)
{


	u64 starttime = getTime();

	int parts = d_tops.size();
	int total = 0, num;
	for (int i = 0; i < parts; i++)
	{
		num = d_tops[i];
		total += num;
	}
	thrust::pair<device_vector<int>::iterator, device_vector<int>::iterator> minmax =
			thrust::minmax_element(d_search.begin(), d_search.end());
	host_vector<int> h_end_index(parts);
	device_vector<int> d_end_index(parts);
	int number_of_each = d_search.size() / parts;
	for (int i = 0; i < parts; i++)
	{
		h_end_index[i] = (i + 1) * number_of_each;
	}
	d_end_index = h_end_index;
	d_top_indexes.clear(), d_top_indexes.resize(total);

	ValueOfInt val;
	val.max = *minmax.second;
	float min = *minmax.first;
	float max = *minmax.second;
	bucket_topk<int, ValueOfInt>(&d_search, val, min, max, &d_tops,
			&d_end_index, parts, &d_top_indexes);

	u64 endtime = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: Selection  takes %f ms (GPU k selection)<<<<",
			getInterval(starttime, endtime));
}

void GPUGenie::topk(device_vector<float>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes)
{
	u64 starttime = getTime();

	int parts = d_tops.size();
	int total = 0, i, num;
	for (i = 0; i < parts; i++)
	{
		num = d_tops[i];
		total += num;
	}
	thrust::pair<device_vector<float>::iterator, device_vector<float>::iterator> minmax =
			thrust::minmax_element(d_search.begin(), d_search.end());
	host_vector<int> h_end_index(parts);
	device_vector<int> d_end_index(parts);

	int number_of_each = d_search.size() / parts;
	for (i = 0; i < parts; i++)
	{
		h_end_index[i] = (i + 1) * number_of_each;
	}
	d_end_index = h_end_index;
	d_top_indexes.clear(), d_top_indexes.resize(total);
	ValueOfFloat val;
	val.max = *minmax.second;
	bucket_topk<float, ValueOfFloat>(&d_search, val, *minmax.first,
			*minmax.second, &d_tops, &d_end_index, parts, &d_top_indexes);

	u64 endtime = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: Selection takes %f ms (GPU k selection)<<<<",
			getInterval(starttime, endtime));
}

void GPUGenie::topk(device_vector<data_t>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes, float dim)
{
	u64 starttime = getTime();

	if (d_tops.size() == 0)
	{
		printf("Error: No query found! Program aborted...");
		exit(1);
	}
	int parts = d_tops.size();
	int total = 0, i, num;
	float minval = 0.0f, maxval = dim;
	for (i = 0; i < parts; i++)
	{
		num = d_tops[i];
		total += num;
	}

	host_vector<int> h_end_index(parts);
	device_vector<int> d_end_index(parts);

	int number_of_each = d_search.size() / parts;
	for (i = 0; i < parts; i++)
	{
		h_end_index[i] = (i + 1) * number_of_each;
	}
	d_end_index = h_end_index;
	d_top_indexes.clear(), d_top_indexes.resize(total);
	ValueOfData val;
	val.max = maxval;
	bucket_topk<data_t, ValueOfData>(&d_search, val, minval, maxval, &d_tops,
			&d_end_index, parts, &d_top_indexes);

	u64 endtime = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: Selection takes %f ms (GPU k selection)<<<<",
			getInterval(starttime, endtime));
}
void GPUGenie::topk(device_vector<u32>& d_search, device_vector<int>& d_tops,
		device_vector<int>& d_top_indexes, u32 dim)
{

	u64 starttime = getTime();

	if (d_tops.size() == 0)
	{
		Logger::log(Logger::ALERT, "Error: No query found! Program aborted...");
		exit(1);
	}
	int parts = d_tops.size();
	int total = 0, i, num;
	u32 minval = 0u, maxval = dim;
	for (i = 0; i < parts; i++)
	{
		num = d_tops[i];
		total += num;
	}

	host_vector<int> h_end_index(parts);
	device_vector<int> d_end_index(parts);

	int number_of_each = d_search.size() / parts;
	for (i = 0; i < parts; i++)
	{
		h_end_index[i] = (i + 1) * number_of_each;
	}
	d_end_index = h_end_index;
	d_top_indexes.clear(), d_top_indexes.resize(total);
	ValueOfU32 val;
	val.max = maxval;
	bucket_topk<u32, ValueOfU32>(&d_search, val, minval, maxval, &d_tops,
			&d_end_index, parts, &d_top_indexes);

	u64 endtime = getTime();
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: Selection takes %f ms (GPU k selection)<<<<",
			getInterval(starttime, endtime));
}
