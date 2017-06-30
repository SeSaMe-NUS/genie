#ifndef GENIE_EXCEPTION_EXCEPTION_H_
#define GENIE_EXCEPTION_EXCEPTION_H_

#include <stdexcept>

namespace genie {
namespace exception {

class NotImplementedException : public std::logic_error {
	public:
		NotImplementedException();
};

} // end of namespace genie::exception
} // end of namespace genie

#endif
