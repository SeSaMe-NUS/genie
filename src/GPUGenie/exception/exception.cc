#include "exception.h"

using namespace genie::exception;
using namespace std;

NotImplementedException::NotImplementedException()
	: logic_error("Function is not implemented")
{}
