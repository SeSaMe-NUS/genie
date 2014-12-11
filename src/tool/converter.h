#ifndef GaLG_tool_converter_s2_h
#define GaLG_tool_converter_s2_h

#include <string>

namespace GaLG {
  namespace tool {
    using namespace std;
    
    int s2i(string&);
    int s2i(string&, void*);

    float s2f(string&);
    float s2f(string&, void*);

    double s2d(string&);
    double s2d(string&, void*);
  }
}

#endif