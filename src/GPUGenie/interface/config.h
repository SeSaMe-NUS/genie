#ifndef GENIE_INTERFACE_CONFIG_H_
#define GENIE_INTERFACE_CONFIG_H_

#include <iostream>
#include <string>

namespace genie {

class Config {
	private:
		uint32_t k_;
		uint32_t query_range_;
		uint32_t num_of_query_;
		bool save_to_gpu_ = false;
		uint8_t gpu_id_ = 0;
		// helper variables
		bool k_initialized_ = false;
		bool query_range_initialized_ = false;
		bool num_of_query_initialized_ = false;
	public:
		uint32_t GetK();
		uint32_t GetQueryRange();
		uint32_t GetNumOfQuery();
		bool GetSaveToGpu();
		uint8_t GetGpuId();
		Config& SetK(uint32_t k);
		Config& SetQueryRange(uint32_t query_range);
		Config& SetNumOfQuery(uint32_t num_of_query);
		Config& SetSaveToGpu(bool save_to_gpu);
		Config& SetGpuId(uint8_t gpu_id);
		Config& LoadFromFile(std::string& filename);
		// helper functions
		bool IsKSet();
		bool IsQueryRangeSet();
		bool IsNumOfQuerySet();
		void Validate();
};

} // end of namespace genie

#endif
