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
		virtual std::shared_ptr<GPUGenie::inv_table> BuildTable(
			const genie::TableData& table_data) = 0;
		virtual std::vector<GPUGenie::query> BuildQuery(
			const std::shared_ptr<const GPUGenie::inv_table>& table,
			const genie::QueryData& query_data) = 0;
		virtual genie::SearchResult Match(
			const std::shared_ptr<const GPUGenie::inv_table>& table,
			const std::vector<GPUGenie::query>& queries) = 0;
		virtual genie::SearchResult Match(
			const std::vector<std::shared_ptr<const GPUGenie::inv_table> >& tables,
			const std::vector<std::vector<GPUGenie::query> >& queries) = 0;
		virtual void Validate();
		void SetK(const uint32_t k);
		void SetNumOfQuery(const uint32_t num_of_query);
		uint32_t GetNumOfQuery() const;
};

class ExecutionPolicyFactory {
	public:
		static std::shared_ptr<ExecutionPolicy> MakePolicy(const Config& config);
};

} // end of namespace genie

#endif
