#ifndef GaLG_tool_converter_h
#define GaLG_tool_converter_h

#include <string>

using namespace std;

namespace GaLG {
  namespace tool {
    int s2i(string&);
    int s2i(string&, void*);

    float s2f(string&);
    float s2f(string&, void*);

    double s2d(string&);
    double s2d(string&, void*);
  }
}

#endif