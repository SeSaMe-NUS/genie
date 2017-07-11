/*! \file genie_errors.h
 *  \brief All the exceptions
 *
 */

#ifndef GENIE_UTILITY_CUDAMACROS_H_
#define GENIE_UTILITY_CUDAMACROS_H_

#include <stdio.h>
#include <genie/exception/exception.h>
#include <genie/utility/Logger.h>

/*! \fn cudaCheckErrors( err )
 *  \brief The wrapper function to validate CUDA calls.
 *
 *  Please wrap all CUDA calls with this function.
 *
 *  Once a CUDA error is detected, a genie::exception::gpu_runtime_error will be
 *  thrown and the error will be logged.
 *
 *	\param err The CUDA error.
 */
#define cudaCheckErrors( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR( err ) __cudaSafeCall( err, __FILE__, __LINE__ )


/*! \fn cudaCheckErrors( err )
 *  \brief Check for existing cuda errors
 *
 *  Once a CUDA error is detected, a genie::exception::gpu_runtime_error will be
 *  thrown and the error will be logged.
 */
#define CUDA_LAST_ERROR() __cudaGetLastError (__FILE__, __LINE__)

/*! \fn inline void __cudaSafeCall(cudaError err, const char *file, const int line)
 *  \brief The hidden wrapper function to validate CUDA calls.
 *
 *  This function should not be called directly. It is wrapped in a
 *  macro expansion cudaCheckErrors( err ) .
 *  Please wrap all CUDA calls with cudaCheckErrors.
 *
 *  Once cuda errors are detected, a genie::exception::gpu_runtime_error will be
 *  thrown and the errors will be logged.
 *
 *	\param err The CUDA error.
 *	\param file The source code file name.
 *	\param line The line number that indicates the source of the error.
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{

	if (cudaSuccess != err)
	{
		char errstr[1000];
		snprintf(errstr, 1000, "cudaSafeCall() failed at %s:%i : %s", file, line, cudaGetErrorString(err));
		genie::utility::Logger::log(genie::utility::Logger::ALERT, "%s", errstr);
		throw(genie::exception::gpu_runtime_error(errstr));
	}

	return;
}


inline void __cudaGetLastError(const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
    	char errstr[1000];
        snprintf(errstr, 1000, "cudaGetLastError() failed at %s:%i : ERR %d - %s.\n",
        	file, line, (int)err, cudaGetErrorString(err));
        genie::utility::Logger::log(genie::utility::Logger::ALERT, "%s", errstr);
		throw(genie::exception::gpu_runtime_error(errstr));
    }
}

#endif

