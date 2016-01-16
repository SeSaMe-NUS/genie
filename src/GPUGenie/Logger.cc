/*
 * Logger.cc
 *
 *  Created on: Jan 4, 2016
 *      Author: luanwenhao
 */

#include "Logger.h"
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <ctime>

const char * const Logger::LEVEL_NAMES[] = {"NONE   ",
											"ALERT  ",
											"INFO   ",
											"VERBOSE",
											"DEBUG  "};

const char * Logger::default_name = "GPUGENIE_LOG.log";
Logger * Logger::logger = NULL;

Logger::Logger(int level)
{
	log_level = level;
	strcpy(logfile_name, default_name);
	logfile = fopen(logfile_name, "a");
}

Logger::~Logger()
{
	fclose(logfile);
}

void Logger::exit(void)
{
	if(logger != NULL)
	{
		log(VERBOSE,"---------Exiting  Logger----------");
		delete logger;
	}

}

Logger* Logger::_logger(void)
{
	if(logger == NULL)
	{
		logger = new Logger(INFO);
		log(VERBOSE,"---------Starting Logger %s----------", logger->logfile_name);

	}
	return logger;
}

void Logger::set_level(int level)
{
	_logger()->log_level = level;
}
int Logger::get_level()
{
	return _logger()->log_level;
}

void Logger::set_logfile_name(const char * name)
{
	if(strcmp(name, _logger()->logfile_name) != 0)
	{
		strcpy(_logger()->logfile_name, name);
		if(logger != NULL)
		{
			fclose(logger->logfile);
			logger = NULL;
			_logger();
		}

	}

}

const char * Logger::get_logfile_name()
{
	return _logger()->logfile_name;
}

int Logger::log(int level, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    timeval curTime;
    gettimeofday(&curTime, NULL);
    int milli = curTime.tv_usec / 1000;

    char buffer [80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

    char currentTime[84];
    sprintf(currentTime, "[%s:%03d %s] ", buffer, milli,LEVEL_NAMES[level]);
    fprintf(_logger()->logger->logfile, currentTime);

    char message[1024];
    vsprintf(message, fmt, args);
    va_end(args);

    fprintf(_logger()->logger->logfile, message);
    fprintf(_logger()->logger->logfile, "\n");

    if(_logger()->logger->log_level >= level)
    {
    	printf(message);
    	printf("\n");
    	return 1;
    } else {
    	return 0;
    }
}
