/*! \file Logger.h
 *  \brief Record run-time information
 *
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#define LOGGER_MAX_FILE_NAME_SIZE 256

#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace GPUGenie
{
	struct query;
	struct inv_table;
	struct inv_compr_table;
}

/*! \class Logger
 *  \brief A utility class to record logs into disk files.
 *
 *  The Logger class is implemented using Singleton model and
 *  all logging calls should go through static method
 *  Logger::log .
 *
 *  Please set your screen print level of log information by
 *  calling Logger::set_level .
 *
 *  Please remember to call the Logger::exit method to safely
 *  instruct the Logger to flush and close all opened files and
 *  free occupied memory.
 *
 */
class Logger
{
public:

	static const int NONE = 0;/*!< Screen print level - no message*/

	static const int ALERT = 1;/*!< Screen print level - only error alters */

	static const int INFO = 2;/*!< Screen print level - progress information */

	static const int VERBOSE = 3;/*!< Screen print level - detailed information */

	static const int DEBUG = 4;/*!< Screen print level - debugging information */

	static const char * const LEVEL_NAMES[];



    /*! \fn virtual ~Logger()
     *  \brief Destructor.
     *
     */
	virtual ~Logger();

	/*! \fn static void exit(void)
	 *  \brief Safely exit the logger.
	 *
	 *  This function should be called if the Logger class is used.
	 */
	static void exit(void);

	/*! \fn static int log(int level, const char *fmt, ...)
	 *  \brief Record and print a message into log file.
	 *
	 *  The log call will direct the parameters except the level to
	 *  std::sprintf .
	 *
	 *  If the current screen print level is higher than the
	 *  provided logging level in the parameter, the message will be
	 *  logged and printed. Otherwise the message will only be
	 *  logged.
	 *
	 *  \param level The logging level of the message.
	 *  \param fmt The printing format of the message. See std::printf .
	 *  \param ... The rest of parameters according to your format.
	 *
	 *  \return 1 if the message is to be printed on screen or 0 otherwise.
	 */
	static int log(int level, const char *fmt, ...);

	static void logQueries(int level, std::vector<GPUGenie::query> &queries, size_t max_print_len = 128);

	static void logResults(int level, std::vector<GPUGenie::query> &queries, std::vector<int> &result,
		std::vector<int> &result_count, size_t max_print_len = 128);

	static void logTable(int level, GPUGenie::inv_table *table, size_t max_print_len = 32);

	static void logInvLists(int level, const std::vector<std::vector<uint32_t> > &rawInvertedLists,
		size_t max_print_len = 128);

	/*! \fn static void set_level(int level)
	 *  \brief Set the screen print level for the logger
	 *
	 *  Once set, the Logger will only print messages whose logging levels
	 *  are higher than the set value to screen.
	 *
	 *  It is recommended to set the screen print level at the
	 *  beginning of the program.
	 *
	 *  \param level The desired screen print level.
	 */
	static void set_level(int level);

	/*! \fn static int get_level()
	 *  \brief Get the current screen print level.
	 *
	 *  \return The current screen print level set in the Logger.
	 */
	static int get_level();

	/*! \fn static void set_logfile_name(const char *name)
	 *  \brief Set the log file name.
	 *
	 *  Once set, the Logger will write all log all messages in the
	 *  current session to the file with the provided name.
	 *
	 *  The file will be created if it does not exist.
	 *
	 *  If not called, the default file name is set using the
	 *  system current date and time.
	 *
	 *  \param name The desired log file name.
	 */
	static void set_logfile_name(const char *name);

	/*! \fn static const char * get_logfile_name()
	 *  \brief Get the current log file name.
	 *
	 *  \return The current log file name set in the Logger.
	 */
	static const char * get_logfile_name();

private:

	/*!
	 * \brief The only available logger instance.
	 */
	static Logger * logger;

	/*!
	 * \brief The current screen print log level.
	 */
	int log_level;

	/*!
	 * \brief The current log file name.
	 */
	char logfile_name[LOGGER_MAX_FILE_NAME_SIZE];

	/*!
	 * \brief The file pointer to the log file.
	 */
	FILE * logfile;

	/*! \fn Logger(int level)
	 *  \brief The constructor for Logger.
	 *
	 *  Attention! This constructor should NOT be called
	 *  in code outside the class.
	 *
	 *	\param level The screen print log level to be set.
	 */
	Logger(int level);

	/*! \fn static Logger* _logger(void)
	 *  \brief Retrieve the only logger instance.
	 *
	 *  Retrieve the only logger instance if it exists,
	 *  or create one if it does not exist.
	 *
	 *  Attention! This constructor should NOT be called
	 *  in code outside the class.
	 *
	 *	\return The only Logger instance.
	 */
	static Logger* _logger(void);
};

#endif /* LOGGER_H_ */
