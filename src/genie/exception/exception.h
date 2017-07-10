#ifndef GENIE_EXCEPTION_EXCEPTION_H_
#define GENIE_EXCEPTION_EXCEPTION_H_

#include <stdexcept>
#include <string>

namespace genie {
namespace exception {

class NotImplementedException : public std::logic_error {
	public:
		NotImplementedException();
};

class InvalidConfigurationException : public std::logic_error {
	public:
		InvalidConfigurationException(const std::string& what);
};

} // end of namespace genie::exception
} // end of namespace genie

#endif
