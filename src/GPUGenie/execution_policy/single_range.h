#ifndef GENIE_EXECUTION_POLICY_SINGLE_RANGE_H_
#define GENIE_EXECUTION_POLICY_SINGLE_RANGE_H_

#include <GPUGenie/interface/execution_policy.h>
#include <GPUGenie/interface/types.h>
#include <GPUGenie.h>

namespace genie {
namespace execution_policy {

class SingleRangeExecutionPolicy : public genie::ExecutionPolicy {
	friend std::shared_ptr<genie::ExecutionPolicy> genie::MakePolicy(const genie::Config& config);
	private:
		uint32_t query_range_;
		SingleRangeExecutionPolicy() = default;
	public:
		virtual std::shared_ptr<GPUGenie::inv_table> BuildTable(
				const genie::TableData& table_data) override final;
		virtual std::vector<GPUGenie::query> BuildQuery(
				const std::shared_ptr<const GPUGenie::inv_table>& table,
				const genie::QueryData& query_data) override final;
		virtual genie::SearchResult Match(
				const std::shared_ptr<const GPUGenie::inv_table>& table,
				const std::vector<GPUGenie::query>& queries) override final;
		virtual genie::SearchResult Match(
				const std::vector<std::shared_ptr<const GPUGenie::inv_table> >& table,
				const std::vector<std::vector<GPUGenie::query> >& queries) override final;
		virtual void Validate() override final;
		void SetQueryRange(const uint32_t query_range);
};

}
}

#endif
