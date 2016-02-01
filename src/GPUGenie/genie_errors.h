/*
 * genie_errors.h
 *
 *  Created on: Feb 1, 2016
 *      Author: luanwenhao
 */

#ifndef GENIE_ERRORS_H_
#define GENIE_ERRORS_H_

#include <stdexcept>
#include <stdio.h>
#include "Logger.h"

#define cudaCheckErrors( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

namespace GPUGenie
{

class genie_error : public std::runtime_error
{
public:
	genie_error(const char * msg);
};

class gpu_bad_alloc : public genie_error
{
public:
	gpu_bad_alloc(const char * msg):genie_error(msg){}

};

class gpu_runtime_error : public genie_error
{
public:
	gpu_runtime_error(const char * msg):genie_error(msg){}

};

class cpu_bad_alloc : public genie_error
{
public:
	cpu_bad_alloc(const char * msg):genie_error(msg){}

};

class cpu_runtime_error : public genie_error
{
public:
	cpu_runtime_error(const char * msg):genie_error(msg){}

};
}

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{

	if (cudaSuccess != err)
	{
		char errstr[1000];
		sprintf(errstr, "cudaSafeCall() failed at %s:%i : %s", file, line,
				cudaGetErrorString(err));
		Logger::log(Logger::ALERT, "%s", errstr);
		throw(GPUGenie::gpu_runtime_error(errstr));
	}

	return;
}

#endif /* GENIE_ERRORS_H_ */
