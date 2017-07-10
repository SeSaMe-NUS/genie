#ifndef GENIE_EXECUTION_POLICY_VALIDATION_H_
#define GENIE_EXECUTION_POLICY_VALIDATION_H_

#include <cstdint>

namespace genie {
namespace execution_policy {
namespace validation {

void ValidateK(uint32_t k);
void ValidateNumOfQueries(uint32_t num_of_queries);
void ValidateQueryRange(uint32_t query_range);

}
}
}

#endif
