#include "converter.h"

#include <cstdlib>
#include <string>

using namespace std;

int GaLG::tool::s2i(string& s)
{
  return atoi(s.c_str());
}
int GaLG::tool::s2i(string& s, void* d)
{
  return s2i(s);
}

float GaLG::tool::s2f(string& s)
{
  return atof(s.c_str());
}
float GaLG::tool::s2f(string& s, void* d)
{
  return s2f(s);
}

double GaLG::tool::s2d(string& s)
{
  return atof(s.c_str());
}
double GaLG::tool::s2d(string& s, void* d)
{
  return s2d(s);  
}