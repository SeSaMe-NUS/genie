#include "GaLG.h"

#include <assert.h>
#include <vector>
#include <string>
using namespace std;

using namespace GaLG;

int main()
{
  raw_data data;
  tool::csv("../../static/countrylist.csv", data);

  data.set_meta(10, "New");
  assert(data.get_meta(10) -> compare("New") == 0);
  data.set_meta("New", "Old");
  assert(data.get_meta(10) -> compare("Old") == 0);

  vector<string> order;
  data.select("Sort Order", order);
  assert(order[0] == "1.1");

  data.select("Common Name", order);
  assert(order[0] == "Afghanistan");

  return 0;
}