#include  "GPUGenie.h" //for ide: to revert it as system file later, change "GPUGenie.h" to "../src/GPUGenie.h"

#include <assert.h>
#include <vector>
#include <string>

using namespace std;
using namespace GPUGenie;

int main()//for ide: from main to main2
{
  raw_data data;
  parser::csv("../../static/countrylist.csv", data);

  vector<string>* re;
  re = data.col("Sort Order");
  assert((*re)[0] == "1.1");

  re = data.col("Common Name");
  assert((*re)[0] == "Afghanistan");

  return 0;
}
