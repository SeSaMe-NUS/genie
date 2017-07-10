#ifndef GENIE_INTERFACE_EXECUTION_POLICY_H_
#define GENIE_INTERFACE_EXECUTION_POLICY_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <genie/GPUGenie.h>
#include "types.h"
#include "config.h"

namespace genie {

/*!
 * \brief ExecutionPolicy class is the interface for building table, building query,
 *        and matching.
 *
 * This is the base class for all the concrete execution policies. Different types of
 * search are implemented by different subclasses of the ExecutionPolicy class.
 */
class ExecutionPolicy {
	protected:
		uint32_t k_;
		uint32_t dim_;
		uint32_t num_of_queries_;
	public:
		/*!
		 * \brief Builds the inverted index with the given data.
		 */
		virtual std::shared_ptr<genie::table::inv_table> BuildTable(
			const genie::TableData& table_data) = 0;
		/*!
		 * \brief Builds the queries with the given query data.
		 */
		virtual std::vector<genie::query::Query> BuildQuery(
			const std::shared_ptr<const genie::table::inv_table>& table,
			const genie::QueryData& query_data) = 0;
		/*!
		 * \brief Match the given queries on the table.
		 */
		virtual genie::SearchResult Match(
			const std::shared_ptr<const genie::table::inv_table>& table,
			const std::vector<genie::query::Query>& queries) = 0;
		/*!
		 * \brief Batched matching with multiple tables and queries for those tables.
		 */
		virtual genie::SearchResult Match(
			const std::vector<std::shared_ptr<const genie::table::inv_table> >& tables,
			const std::vector<std::vector<genie::query::Query> >& queries) = 0;
		/*!
		 * \brief Checks whether the given search parameters are in valid range
		 */
		virtual void Validate();
		/*!
		 * \brief Sets K.
		 */
		void SetK(const uint32_t k);
		/*!
		 * \brief Sets the number of query.
		 */
		void SetNumOfQueries(const uint32_t num_of_queries);
		/*!
		 * \brief Returns the number of query.
		 */
		uint32_t GetNumOfQueries() const;
};

/*!
 * \brief Builds and returns a policy according to the configurations.
 */
std::shared_ptr<ExecutionPolicy> MakePolicy(const Config& config);

} // end of namespace genie

#endif
