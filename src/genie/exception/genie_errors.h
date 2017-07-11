/*! \file genie_errors.h
 *  \brief All the exceptions
 *
 */

#ifndef GENIE_ERRORS_H_
#define GENIE_ERRORS_H_

#include <stdexcept>
#include <stdio.h>
#include <genie/utility/Logger.h>

namespace genie {
namespace exception {

/*! \class genie_error
 *  \brief The base class for genie error exceptions
 *
 *  The genie error exception class is inherited from std::runtime_error.
 */
class genie_error : public std::runtime_error
{
public:
	/*! \fn genie_error(const char * msg)
	 *  \brief The constructor for genie_error.
	 *	\param msg The error message.
	 */
	genie_error(const char * msg);
};

/*! \class gpu_bad_alloc
 *  \brief The error class for memory exceptions occurred on GPU.
 *
 *  The gpu_bad_alloc error class is inherited from genie::exception::genie_error.
 *  All memory exceptions on GPU will be casted to gpu_bad_alloc and
 *  the error message will be kept.
 *
 *  \ref genie::exception::genie_error
 */
class gpu_bad_alloc : public genie_error
{
public:
	/*! \fn gpu_bad_alloc(const char * msg)
	 *  \brief The constructor for gpu_bad_alloc.
	 *	\param msg The error message.
	 */
	gpu_bad_alloc(const char * msg):genie_error(msg){}

};

/*! \class gpu_runtime_error
 *  \brief The error class for runtime exceptions occurred on GPU.
 *
 *  The gpu_runtime_error error class is inherited from genie::exception::genie_error.
 *  All runtime exceptions on GPU will be casted to gpu_runtime_error and
 *  the error message will be kept.
 *
 *  \ref genie::exception::genie_error
 */
class gpu_runtime_error : public genie_error
{
public:
	/*! \fn gpu_runtime_error(const char * msg)
	 *  \brief The constructor for gpu_runtime_error.
	 *	\param msg The error message.
	 */
	gpu_runtime_error(const char * msg):genie_error(msg){}

};

/*! \class cpu_bad_alloc
 *  \brief The error class for memory exceptions occurred on CPU.
 *
 *  The cpu_bad_alloc error class is inherited from genie::exception::genie_error.
 *  All memory exceptions on CPU will be casted to cpu_bad_alloc and
 *  the error message will be kept.
 *
 *  \ref genie::exception::genie_error
 */
class cpu_bad_alloc : public genie_error
{
public:
	/*! \fn cpu_bad_alloc(const char * msg)
	 *  \brief The constructor for cpu_bad_alloc.
	 *	\param msg The error message.
	 */
	cpu_bad_alloc(const char * msg):genie_error(msg){}

};

/*! \class cpu_runtime_error
 *  \brief The error class for runtime exceptions occurred on CPU.
 *
 *  The cpu_runtime_error error class is inherited from genie::exception::genie_error.
 *  All runtime exceptions on CPU will be casted to cpu_runtime_error and
 *  the error message will be kept.
 *
 *  \ref genie::exception::genie_error
 */
class cpu_runtime_error : public genie_error
{
public:
	/*! \fn cpu_runtime_error(const char * msg)
	 *  \brief The constructor for cpu_runtime_error.
	 *	\param msg The error message.
	 */
	cpu_runtime_error(const char * msg):genie_error(msg){}

};

} // namespace exception
} // namespace genie


#endif

