#ifndef GENIE_EXCEPTION_EXCEPTION_H_
#define GENIE_EXCEPTION_EXCEPTION_H_

#include <stdexcept>
#include <string>

namespace genie {
namespace exception {

class NotImplementedException : public std::logic_error
{
	public:
		NotImplementedException() : std::logic_error("Function is not implemented") {}
};

class InvalidConfigurationException : public std::logic_error
{
	public:
		InvalidConfigurationException(const std::string& what) : std::logic_error(what) {}
};

class gpu_bad_alloc : public std::runtime_error
{
    public:
        gpu_bad_alloc(const std::string& what) : std::runtime_error(what) {}
};

class gpu_runtime_error : public std::runtime_error
{
    public:
        gpu_runtime_error(const std::string& what) : std::runtime_error(what) {}
};

class cpu_bad_alloc : public std::runtime_error
{
public:
    cpu_bad_alloc(const std::string& what) : std::runtime_error(what) {}

};

class cpu_runtime_error : public std::runtime_error
{
public:
    cpu_runtime_error(const std::string& what) : std::runtime_error(what) {}

};

} // end of namespace genie::exception
} // end of namespace genie

#endif
