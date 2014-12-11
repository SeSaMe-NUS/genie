#include "GaLG.h"

#include <assert.h>
#include <vector>
#include <string>
using namespace std;

using namespace GaLG;

int main()
{
  container::raw_data data;
  tool::parser::csv("../../static/countrylist.csv", data);

  vector<string> order;
  data.select("Sort Order", order);
  assert(order[0] == "1.1");

  data.select("Common Name", order);
  assert(order[0] == "Afghanistan");

  return 0;
}