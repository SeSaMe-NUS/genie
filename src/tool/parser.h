#ifndef GaLG_tool_parser_h
#define GaLG_tool_parser_h

#include "container/raw_data.h"

namespace GaLG {
  namespace tool {
    using namespace container;
    int csv(string file, raw_data& data);
  }
}

#endif
