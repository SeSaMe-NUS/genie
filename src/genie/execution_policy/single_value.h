#ifndef GENIE_EXECUTION_POLICY_SINGLE_VALUE_H_
#define GENIE_EXECUTION_POLICY_SINGLE_VALUE_H_

#include <genie/interface/execution_policy.h>
#include <genie/interface/types.h>
#include <genie/GPUGenie.h>

namespace genie {
namespace execution_policy {

class SingleValueExecutionPolicy : public genie::ExecutionPolicy {
	friend std::shared_ptr<genie::ExecutionPolicy> genie::MakePolicy(const genie::Config& config);
	private:
		SingleValueExecutionPolicy() = default;
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
};

}
}

#endif
