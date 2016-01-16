/*
 * Timing.cc
 *
 *  Created on: Jan 16, 2016
 *      Author: luanwenhao
 */

#include <sys/time.h>
#include <stdlib.h>
#include "Timing.h"

unsigned long long getTime()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	unsigned long long ret = tv.tv_usec;

	/* Adds the seconds after converting them to microseconds (10^-6) */
	ret += (tv.tv_sec * 1000 * 1000);

	return ret;
}

double getInterval(unsigned long long start, unsigned long long stop)
{
	return (double)(stop - start) / 1000.0;
}
