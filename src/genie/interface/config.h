#ifndef GENIE_INTERFACE_CONFIG_H_
#define GENIE_INTERFACE_CONFIG_H_

#include <iostream>
#include <string>

namespace genie {

/*!
 * \brief Config class holds all user configurable settings of GENIE
 *
 * The required options are:
 * - K
 * - Number of query
 *
 * The following options have default values:
 * - Save to gpu = false
 * - GPU ID = 0
 */
class Config {
	private:
		uint32_t k_;
		uint32_t query_range_;
		uint32_t num_of_queries_;
		bool save_to_gpu_ = false;
		uint8_t gpu_id_ = 0;
		bool k_initialized_ = false;
		bool query_range_initialized_ = false;
		bool num_of_queries_initialized_ = false;
	public:
		/*!
		 * \brief Return K.
		 */
		uint32_t GetK() const;
		/*!
		 * \brief Return the range used in range-based search.
		 */
		uint32_t GetQueryRange() const;
		/*!
		 * \brief Return the number of query to search.
		 */
		uint32_t GetNumOfQueries() const;
		/*!
		 * \brief Return whether table should be saved to GPU after search.
		 */
		bool GetSaveToGpu() const;
		/*!
		 * \brief Return the ID of the GPU used.
		 */
		uint8_t GetGpuId() const;
		/*!
		 * \brief Set k.
		 */
		Config& SetK(const uint32_t k);
		/*!
		 * \brief Set the range used in range-based search.
		 */
		Config& SetQueryRange(const uint32_t query_range);
		/*!
		 * \brief Set the number of query to search.
		 */
		Config& SetNumOfQueries(const uint32_t num_of_queries);
		/*!
		 * \brief Set whether table should be saved to GPU after search.
		 */
		Config& SetSaveToGpu(const bool save_to_gpu);
		/*!
		 * \brief Set the ID of the GPU used.
		 */
		Config& SetGpuId(const uint8_t gpu_id);
		/*!
		 * \brief Load configuration from a file.
		 */
		Config& LoadFromFile(const std::string& filename);
		/*!
		 * \brief Return whether K has been set.
		 */
		bool IsKSet() const;
		/*!
		 * \brief Return whether the query range has been set.
		 */
		bool IsQueryRangeSet() const;
		/*!
		 * \brief Return whether the number of query has been set.
		 */
		bool IsNumOfQueriesSet() const;
		/*!
		 * \brief Validate that all required options are set.
		 */
		void Validate() const;
		/*!
		 * \brief Display enabled configuration options.
		 */
		void DisplayConfiguration();
};

} // end of namespace genie

#endif
