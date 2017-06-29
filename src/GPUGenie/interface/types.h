#ifndef GENIE_INTERFACE_TYPES_H_
#define GENIE_INTERFACE_TYPES_H_

#include <vector>
#include <utility>

namespace genie {

typedef std::vector<std::vector<int> > TableData;
typedef std::vector<std::vector<int> > QueryData;
typedef std::pair<std::vector<int>, std::vector<int> > SearchResult;

}

#endif
