#ifndef GENIE_INTERFACE_TYPES_H_
#define GENIE_INTERFACE_TYPES_H_

#include <vector>
#include <utility>

namespace genie {

/*!
 * \brief Raw table data format used for building the table.
 */
typedef std::vector<std::vector<int> > TableData;
/*!
 * \brief Raw query data format used for building the queries.
 */
typedef std::vector<std::vector<int> > QueryData;
/*!
 * \brief Matching result (top K's ID and count).
 */
typedef std::pair<std::vector<int>, std::vector<int> > SearchResult;

}

#endif
