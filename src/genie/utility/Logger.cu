/*! \file Logger.cc
 *  \brief Implementation for Logger.h
 */
#include <ctime>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include <genie/compression/DeviceCodecFactory.h>
#include <genie/utility/Timing.h>
#include <genie/query/query.h>
#include <genie/table/inv_table.h>
#include <genie/table/inv_compr_table.h>

#include "Logger.h"

using namespace std;
const char * const Logger::LEVEL_NAMES[] =
{ "NONE   ", "ALERT  ", "INFO   ", "VERBOSE", "DEBUG  " };

Logger * Logger::logger = NULL;

Logger::Logger(int level)
{
	log_level = level;
	std::string s = currentDateTime();
	char fout_name[128];
	sprintf(fout_name, "GPUGENIE_LOG-%s.log", s.c_str());
    stringstream ss;
    ss<<"log/"<<string(fout_name);
    struct stat st;
    if(stat("log", &st) == -1)
        mkdir("log", 0700);
	strcpy(logfile_name, ss.str().c_str());
	logfile = fopen(logfile_name, "a");
}

Logger::~Logger()
{
	fclose(logfile);
}

void Logger::exit(void)
{
	if (logger != NULL)
	{
		log(VERBOSE, "---------Exiting  Logger----------");
		delete logger;
	}

}

Logger* Logger::_logger(void)
{
	if (logger == NULL)
	{
		logger = new Logger(INFO);
		log(VERBOSE, "---------Starting Logger %s----------",
				logger->logfile_name);

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
	if (strcmp(name, _logger()->logfile_name) != 0)
	{
		fclose(_logger()->logfile); // close previously opened logger file
		strcpy(_logger()->logfile_name, name);
		_logger()->logfile = fopen(_logger()->logfile_name, "a"); // open a new file for logging
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

	char buffer[80];
	strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

	char currentTime[84];
	sprintf(currentTime, "[%s:%03d %s] ", buffer, milli, LEVEL_NAMES[level]);
	fprintf(_logger()->logger->logfile, "%s", currentTime);

	char message[1024];
	vsprintf(message, fmt, args);
	va_end(args);

	fprintf(_logger()->logger->logfile, "%s", message);
	fprintf(_logger()->logger->logfile, "\n");

	if (_logger()->logger->log_level >= level)
	{
		printf("%s", message);
		printf("\n");
		return 1;
	}
	else
	{
		return 0;
	}
}


void Logger::logQueries(int level, std::vector<GPUGenie::query> &queries, size_t max_print_len)
{
    for (GPUGenie::query &q : queries)
    {
        Logger::log(level, "Query idx: %d, topk: %d, count_ranges: %d, selectivity: %f",
                    q.index(), q.topk(), q.count_ranges(), q.selectivity());
        q.print(max_print_len);

        std::vector<GPUGenie::query::dim> dims;
        q.dump(dims);

        for (GPUGenie::query::dim &d : dims){
            Logger::log(level, "  Dim -- query: %d, order: %d, start_pos: %d, end_pos: %d",
                    d.query, d.order, d.start_pos, d.end_pos);
        }
    }
}

void Logger::logResults(int level, std::vector<GPUGenie::query> &queries, std::vector<int> &result,
		std::vector<int> &result_count, size_t max_print_len)
{
    size_t resultsBeginIdx = 0;
    for (GPUGenie::query &q : queries)
    {
        Logger::log(level, "Query idx: %d, topk: %d, count_ranges: %d, selectivity: %f",
                    q.index(), q.topk(), q.count_ranges(), q.selectivity());

        std::stringstream ss;
        size_t noResultsToPrint = std::min(q.topk(),static_cast<int>(max_print_len));
        for (size_t i = 0; i < noResultsToPrint; ++i)
            ss << result[resultsBeginIdx+i] << "(" << result_count[resultsBeginIdx+i] << ") ";
        Logger::log(level, "  Results: %s", ss.str().c_str());
        resultsBeginIdx += q.topk();
    }
}

void Logger::logTable(int level, GPUGenie::inv_table *table, size_t max_print_len)
{
    if (table->build_status() == GPUGenie::inv_table::not_builded)
    {
        Logger::log(level, "Inv table not built.");
        return;
    }
    #ifdef GENIE_COMPR 
        GPUGenie::inv_compr_table* comprTable = dynamic_cast<GPUGenie::inv_compr_table*>(table);
        if (comprTable)
        	Logger::log(level,"Compressed table: %s",
                GPUGenie::DeviceCodecFactory::getCompressionName(comprTable->getCompression()).c_str());
        else
        	Logger::log(level,"Plain table: ");
    #endif

    std::stringstream ss;    
    std::vector<int> *ck = table->ck();
    if (ck)
    {
        auto end = (ck->size() <= max_print_len) ? ck->end() : (ck->begin() + max_print_len); 
        std::copy(ck->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(level, "  CK:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }
    std::vector<int> *inv_pos = 
        #ifdef GENIE_COMPR 
            comprTable ? comprTable->uncompressedInvPos() :
        #endif
        table->inv_pos();
    if (inv_pos)
    {
        auto end = (inv_pos->size() <= max_print_len) ? inv_pos->end() : (inv_pos->begin() + max_print_len); 
        std::copy(inv_pos->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(level, "  INV_POS:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }
    std::vector<int> *inv =
        #ifdef GENIE_COMPR 
            comprTable ? comprTable->uncompressedInv() :
        #endif
        table->inv();
    if (inv)
    {
        auto end = (inv->size() <= max_print_len) ? inv->end() : (inv->begin() + max_print_len); 
        std::copy(inv->begin(), end, std::ostream_iterator<int>(ss, " "));
        Logger::log(level, "  INV:\n %s", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }
    #ifdef GENIE_COMPR 
    if (comprTable)
    {
    	std::vector<int> *compr_inv_pos = comprTable->inv_pos();
	    if (compr_inv_pos)
	    {
	        auto end = (compr_inv_pos->size() <= max_print_len) ?
	        					compr_inv_pos->end() : (compr_inv_pos->begin() + max_print_len); 
	        std::copy(compr_inv_pos->begin(), end, std::ostream_iterator<int>(ss, " "));
	        Logger::log(level, "  COMPR_INV_POS:\n %s", ss.str().c_str());
	        ss.str(std::string());
	        ss.clear();
	    }
    	std::vector<uint32_t> *compr_inv = comprTable->compressedInv();
	    if (compr_inv)
	    {
	        auto end = (compr_inv->size() <= max_print_len) ? compr_inv->end() : (compr_inv->begin() + max_print_len);
	        int cnt = 0;
	        for (auto it = compr_inv->begin(); it != end; it++){
	        	ss << std::hex << std::setfill('0') << std::setw(8) << *it << " ";
	        	if (cnt % 6 == 5)
	        		ss << std::endl;
	        	cnt++;
	        }
	        Logger::log(level, "  COMPR_INV:\n%s", ss.str().c_str());
	        ss.str(std::string());
	        ss.clear();
	    }
    }
    #endif
}

void Logger::logInvLists(int level, const std::vector<std::vector<uint32_t> > &rawInvertedLists,
		size_t max_print_len)
{
    std::stringstream ss;
    auto inv_it_end = (rawInvertedLists.size() <= max_print_len)
                            ? rawInvertedLists.end() : (rawInvertedLists.begin() + max_print_len);
    Logger::log(level, "rawInvertedLists.size(): %d", rawInvertedLists.size());
    for (auto inv_it = rawInvertedLists.begin(); inv_it != inv_it_end; inv_it++)
    {
        const std::vector<uint32_t> &invList = *inv_it; 
        auto end = (invList.size() <= max_print_len) ? invList.end() : (invList.begin() + max_print_len);
        std::copy(invList.begin(), end, std::ostream_iterator<uint32_t>(ss, " "));
        Logger::log(level, "*** [%s]", ss.str().c_str());
        ss.str(std::string());
        ss.clear();
    }
}
