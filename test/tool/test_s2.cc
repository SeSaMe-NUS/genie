#include "GaLG.h"

#include <assert.h>
#include <string>

using namespace GaLG;
using namespace std;

#define EQUAL(X, Y) ((X-Y) < 0.00001 && (X-Y) > -0.00001)

int main()
{
  string value = "10";
  assert(tool::s2i(value) == 10);
  value = "11";
  assert(tool::s2i(value) == 11);
  value = "0";
  assert(tool::s2i(value) == 0);
  value = "-10";
  assert(tool::s2i(value) == -10);
  value = "0.0";
  assert(tool::s2i(value) == 0);
  value = "0.01";
  assert(tool::s2i(value) == 0);
  value = "1.01";
  assert(tool::s2i(value) == 1);
  value = "abc";
  assert(tool::s2i(value) == 0);

  value = "0.0";
  assert(EQUAL(tool::s2f(value), 0));
  value = "0.01";
  assert(EQUAL(tool::s2f(value), 0.01));
  value = "1.01";
  assert(EQUAL(tool::s2f(value), 1.01));
  value = "abc";
  assert(EQUAL(tool::s2f(value), 0));
}