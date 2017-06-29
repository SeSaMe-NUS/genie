#include <iostream>

#include "execution_policy.h"
#include <GPUGenie/execution_policy/single_value.h>
#include <GPUGenie/execution_policy/single_range.h>
#include <GPUGenie/exception/exception.h>

using namespace genie;
using namespace std;

shared_ptr<ExecutionPolicy> genie::ExecutionPolicyFactory::MakePolicy(Config& config)
{
	shared_ptr<ExecutionPolicy> generated_policy;

	if (0 != config.GetQueryRange())
	{
		shared_ptr<execution_policy::SingleRangeExecutionPolicy> policy = make_shared<execution_policy::SingleRangeExecutionPolicy>();
		policy->SetK(config.GetK());
		policy->SetNumOfQuery(config.GetNumOfQuery());
		policy->SetQueryRange(config.GetQueryRange());
		generated_policy = policy;
	}
	else
	{
		shared_ptr<execution_policy::SingleValueExecutionPolicy> policy = make_shared<execution_policy::SingleValueExecutionPolicy>();
		policy->SetK(config.GetK());
		policy->SetNumOfQuery(config.GetNumOfQuery());
		generated_policy = policy;
	}

	return generated_policy;
}

shared_ptr<GPUGenie::inv_table> genie::ExecutionPolicy::LoadTable(TableData& table_data)
{
	throw genie::exception::NotImplementedException();
}

vector<GPUGenie::query> genie::ExecutionPolicy::LoadQuery(
	shared_ptr<GPUGenie::inv_table>& table,
	QueryData& query_data)
{
	throw genie::exception::NotImplementedException();
}

SearchResult genie::ExecutionPolicy::KnnSearch(
	shared_ptr<GPUGenie::inv_table>& table,
	vector<GPUGenie::query>& queries)
{
	throw genie::exception::NotImplementedException();
}

SearchResult genie::ExecutionPolicy::KnnSearch(
	vector<shared_ptr<GPUGenie::inv_table> >& tables,
	vector<vector<GPUGenie::query> >& queries)
{
	throw genie::exception::NotImplementedException();
}

void genie::ExecutionPolicy::SetK(uint32_t k)
{
	k_ = k;
}

void genie::ExecutionPolicy::SetNumOfQuery(uint32_t num_of_query)
{
	num_of_query_ = num_of_query;
}

uint32_t genie::ExecutionPolicy::GetNumOfQuery()
{
	return num_of_query_;
}
