#include "GPUGenie.h" //for ide: to revert it as system file later, change "GPUGenie.h" to "../../src/GPUGenie.h"

#include <assert.h>

using namespace GPUGenie;

int main()//for ide: from main to main3
{
  raw_data data;
  int lines = parser::csv("../../static/countrylist.csv", data);
  assert(lines == 273);

  assert(data.i_size() == 272);
  assert(data.m_size() == 14);
  assert(data.meta(0) -> compare("Sort Order") == 0);
  assert(data.meta(data.m_size() - 1) -> compare("IANA Country Code TLD") == 0);
  assert((*data.row(0))[0].compare("1.1") == 0);
  assert((*data.row(271))[0].compare("272") == 0);

  return 0;
}
