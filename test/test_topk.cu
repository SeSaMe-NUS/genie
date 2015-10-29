#include "GaLG.h" //for ide: to revert it as system file later, change "GaLG.h" to "../src/GaLG.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <thrust/device_vector.h>

using namespace GaLG;
using namespace thrust;

int
main()//for ide: from main to main10
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

  query q(table);//for ide: comment it
  q.attr(0, 2, 3, 0.3);//for ide:  comment it
  q.attr(1, 2, 4, 0.7);//for ide:  comment it

  device_vector<int> d_c;
  device_vector<float> d_f;

  match(table, q, d_c, d_f);//for ide:  comment it

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
  assert(d_indexes[1] == 6);
  assert(d_indexes[2] == 2);
}
