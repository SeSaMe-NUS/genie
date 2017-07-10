/*! \file genie_errors.cu
 *  \brief Implementation for genie_errors.h
 */

#include <genie/utility/Logger.h>

#include "genie_errors.h"

using namespace genie::utility;

genie::exception::genie_error::genie_error(const char * msg): std::runtime_error(msg){
	Logger::log(Logger::ALERT, "%s", msg);
}





