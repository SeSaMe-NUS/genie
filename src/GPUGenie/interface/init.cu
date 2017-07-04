#include "init.h"
#include <GPUGenie.h>

using namespace genie;

void genie::Init(Config& config)
{
	static bool initialized = false;
	static uint32_t current_gpu_id = 0;
	uint32_t new_gpu_id = config.GetGpuId();
	if (!initialized || new_gpu_id != current_gpu_id)
	{
		int device_count;

		CUDA_CHECK_ERROR(cudaGetDeviceCount(&device_count));
		if (device_count == 0)
		{
			throw GPUGenie::cpu_runtime_error("Nvidia CUDA-supported GPU not found.");
		}
		else if (device_count <= new_gpu_id)
		{
			Logger::log(Logger::INFO,
					"[Info] Device %d not found! Changing to %d...",
					new_gpu_id, 0);
			new_gpu_id = 0;
			config.SetGpuId(0);
		}
		CUDA_CHECK_ERROR(cudaSetDevice(new_gpu_id));
		current_gpu_id = new_gpu_id;

		initialized = true;
	}
}
