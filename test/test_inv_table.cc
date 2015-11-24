#include "GPUGenie.h" //for ide: to revert it as system file later, change "GPUGenie.h" to "../src/GPUGenie.h"
#include <assert.h>
#include <vector>

using namespace GPUGenie;
using namespace std;

int
main()//for ide:from main to main1
{
  raw_data data;
  parser::csv("../../static/t1.csv", data);

  vector<string>* a = data.col(0);
  inv_list list(data.col("g1"));
  inv_table table;
  table.append(list);
  list.invert(data.col("g2"));
  table.append(list);
  table.build();

  vector<int>& inv = *table.inv();
  assert(inv[0] == 0);
  assert(inv[1] == 2);
  assert(inv[2] == 5);
  assert(inv[3] == 8);
  assert(inv[4] == 11);

  vector<int>& ck = *table.ck();

  int shifter = table.shifter();
  int key = (0 << shifter) + 0;
  assert(ck[key] == 5);
  key = (0 << shifter) + 1;
  assert(ck[key] == 8);

  return 0;
}
