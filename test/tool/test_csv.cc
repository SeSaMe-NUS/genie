#include "GaLG.h"

#include <assert.h>

using namespace GaLG;

int main()
{
  container::raw_data data;
  int lines = tool::csv("../../static/countrylist.csv", data);
  assert(lines == 273);

  assert(data.num_of_instances == 272);
  assert(data.num_of_attributes == 14);
  assert(data.meta[0].compare("Sort Order") == 0);
  assert(data.meta[data.meta.size() - 1].compare("IANA Country Code TLD") == 0);
  assert(data.instance[0][0].compare("1.1") == 0);
  assert(data.instance[271][0].compare("272") == 0);

  return 0;
}
