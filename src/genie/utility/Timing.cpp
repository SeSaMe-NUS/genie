/*! \file Timing.cc
 *  \brief Implementation for functions in Timing.h
 */

#include <sys/time.h>
#include <stdlib.h>
#include "Timing.h"

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d-%X", &tstruct);

    return buf;
}

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
	return (double) (stop - start) / 1000.0;
}
