#ifndef GENIE_INTERFACE_CONFIG_H_
#define GENIE_INTERFACE_CONFIG_H_

#include <iostream>
#include <string>

namespace genie {

class Config {
	private:
		uint32_t k_ = 1;
		uint32_t query_range_ = 0;
		uint32_t num_of_query_ = 0;
		bool save_to_gpu_ = false;
		uint8_t gpu_id_ = 0;
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
};

} // end of namespace genie

#endif
