/*! \file Timing.h
 *  \brief Functions about getting system time.
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <string>

/*! \fn std::string currentDataTime()
 *  \brief Get current data time
 *
 *  \return Current date/time, format is YYYY-MM-DD.HH:mm:ss
 */
std::string currentDateTime();

/*! \fn unsigned long long getTime()
 *  \brief Get system time
 *
 *  \return system time
 */
unsigned long long getTime();

/*! \fn double getInterval(unsigned long long start, unsigned long long stop)
 *  \brief Calculate time interval from start to end
 *
 *  \param start Start time
 *  \param stop Stop time
 *
 *  \return time interval(millisecond) between start and stop
 */
double getInterval(unsigned long long start, unsigned long long stop);

#endif /* TIMING_H_ */
