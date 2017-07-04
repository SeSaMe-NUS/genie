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
		uint32_t GetK() const;
		uint32_t GetQueryRange() const;
		uint32_t GetNumOfQuery() const;
		bool GetSaveToGpu() const;
		uint8_t GetGpuId() const;
		Config& SetK(const uint32_t k);
		Config& SetQueryRange(const uint32_t query_range);
		Config& SetNumOfQuery(const uint32_t num_of_query);
		Config& SetSaveToGpu(const bool save_to_gpu);
		Config& SetGpuId(const uint8_t gpu_id);
		Config& LoadFromFile(const std::string& filename);
		// helper functions
		bool IsKSet() const;
		bool IsQueryRangeSet() const;
		bool IsNumOfQuerySet() const;
		void Validate() const;
		void DisplayConfiguration();
};

} // end of namespace genie

#endif
