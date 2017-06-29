#ifndef GENIE_INTERFACE_EXECUTION_POLICY_H_
#define GENIE_INTERFACE_EXECUTION_POLICY_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <GPUGenie.h>
#include "types.h"
#include "config.h"

namespace genie {

class ExecutionPolicy {
	protected:
		uint32_t k_;
		uint32_t dim_;
		uint32_t num_of_query_;
	public:
		virtual std::shared_ptr<GPUGenie::inv_table> LoadTable(
			genie::TableData& table_data);
		virtual std::vector<GPUGenie::query> LoadQuery(
			std::shared_ptr<GPUGenie::inv_table>& table,
			genie::QueryData& query_data);
		virtual genie::SearchResult KnnSearch(
			std::shared_ptr<GPUGenie::inv_table>& table,
			std::vector<GPUGenie::query>& queries);
		virtual genie::SearchResult KnnSearch(
			std::vector<std::shared_ptr<GPUGenie::inv_table> >& tables,
			std::vector<std::vector<GPUGenie::query> >& queries);
		void SetK(uint32_t k);
		void SetNumOfQuery(uint32_t num_of_query);
		uint32_t GetNumOfQuery();
};

class ExecutionPolicyFactory {
	public:
		static std::shared_ptr<ExecutionPolicy> MakePolicy(Config& config);
};

} // end of namespace genie

#endif
