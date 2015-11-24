#include "GPUGenie.h" //for sys: to revert it as system file later, change "GPUGenie.h" to "../src/GPUGenie.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <thrust/device_vector.h>

using namespace GPUGenie;
using namespace thrust;

#define EQUAL(X, Y) ((X-Y<0.0001) && (Y-X <0.0001))

int
main()//for ide: main to main9
{
  srand(time(NULL));
  vector<int> v(100);
  int i;
  for (i = 0; i < 100; i++)
    {
      v[i] = rand() % 100;
    }
  inv_list list(v);
  inv_table table;
  table.append(list);

  for (i = 0; i < 100; i++)
    {
      v[i] = rand() % 100;
    }
  list.invert(v);
  table.append(list);

  query q(table);//for ide: comment it

  q.attr(0, rand() % 100, rand() % 100, 0.3);//for ide: comment it
  q.attr(0, rand() % 100, rand() % 100, 0.7);//for ide: comment it

  device_vector<int> d_c1, d_c2;
  device_vector<float> d_a1, d_a2;

  table.build();
  int size1 = table.ck()->size();
  //match(table, q, d_c1, d_a1);//for ide: comment it

  table.build_compressed();
  int size2 = table.ck()->size();
  match(table, q, d_c2, d_a2);//for ide: comment it

  for (i = 0; i < 100; i++)
    {
      assert(d_c1[i] == d_c2[i]);
      assert(EQUAL(d_a1[i], d_a2[i]));
    }

  /* about 500 times larger */
  printf("%d\n", size1/size2);
  return 0;
}
