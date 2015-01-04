#include "GaLG.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <thrust/device_vector.h>

using namespace GaLG;
using namespace thrust;

int
main()
{
  raw_data data;
  parser::csv("../../static/t1.csv", data);
  inv_list list;
  inv_table table;
  list.invert(data.col("g1"));
  table.append(list);
  list.invert(data.col("g2"));
  table.append(list);
  table.build_compressed();

  query q(table);
  q.attr(0, 2, 3, 0.3);
  q.attr(1, 2, 4, 0.7);

  device_vector<int> d_c;
  device_vector<float> d_f;

  match(table, q, d_c, d_f);

  for (int i = 0; i < d_c.size(); i++)
    {
      int num = d_c[i];
      printf("%d ", num);
    }

  device_vector<int> d_tops(1);
  d_tops[0] = 3;
  device_vector<int> d_indexes;
  topk(d_f, d_tops, d_indexes);

  assert(d_indexes[0] == 1);
  assert(d_indexes[0] == 6);
  assert(d_indexes[0] == 2);
}
