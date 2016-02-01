#include "Logger.h"

#include "genie_errors.h"

GPUGenie::genie_error::genie_error(const char * msg): std::runtime_error(msg){
	Logger::log(Logger::ALERT, "%s", msg);
}





