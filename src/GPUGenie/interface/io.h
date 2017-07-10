#ifndef GENIE_INTERFACE_IO_H_
#define GENIE_INTERFACE_IO_H_

#include <vector>
#include <string>
#include <memory>
#include <GPUGenie.h>
#include <GPUGenie/interface/execution_policy.h>
#include "types.h"

namespace genie {

/*!
 * \brief Reads data from a CSV file and returns in TableData format.
 */
TableData LoadTableDataFromCsv(const std::string& filename);
/*!
 * \brief Reads data from a binary file and returns pre-built table.
 */
std::shared_ptr<GPUGenie::inv_table> LoadTableFromBinary(const std::string& filename);
/*!
 *  \brief Save inverted table to a binary file.
 */
void SaveTableToBinary(const std::string& filename, const std::shared_ptr<const GPUGenie::inv_table> &table);
/*!
 * \brief Reads query from a CSV file and returns in QueryData format.
 */
QueryData LoadQueryDataFromCsv(const std::string& filename, const std::shared_ptr<const ExecutionPolicy>& policy);

} // end of namespace genie

#endif
