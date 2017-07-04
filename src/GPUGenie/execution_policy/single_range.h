#ifndef GENIE_EXECUTION_POLICY_SINGLE_RANGE_H_
#define GENIE_EXECUTION_POLICY_SINGLE_RANGE_H_

#include <GPUGenie/interface/execution_policy.h>
#include <GPUGenie/interface/types.h>
#include <GPUGenie.h>

namespace genie {
namespace execution_policy {

class SingleRangeExecutionPolicy : public genie::ExecutionPolicy {
	friend class genie::ExecutionPolicyFactory;
	private:
		uint32_t query_range_;
		SingleRangeExecutionPolicy() = default;
	public:
		virtual std::shared_ptr<GPUGenie::inv_table> BuildTable(genie::TableData& table_data) override final;
		virtual std::vector<GPUGenie::query> BuildQuery(std::shared_ptr<GPUGenie::inv_table>& table,
				genie::QueryData& query_data) override final;
		virtual genie::SearchResult Match(std::shared_ptr<GPUGenie::inv_table>& table,
				std::vector<GPUGenie::query>& queries) override final;
		virtual genie::SearchResult Match(std::vector<std::shared_ptr<GPUGenie::inv_table> >& table,
				std::vector<std::vector<GPUGenie::query> >& queries) override final;
		virtual void Validate() override final;
		void SetQueryRange(uint32_t query_range);
};

}
}

#endif
