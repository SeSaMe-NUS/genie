#include <GaLG.h>
#include <vector>
#include <string>
#include <stdio.h>

using namespace GaLG;
using namespace std;

int
main()
{
  raw_data data;
  parser::csv("../static/t1.csv", data);

  inv_list list;
  inv_table table;

  list.invert(data.col("g1"));
  table.append(list);

  list.invert(data.col("g2"));
  table.append(list);

  vector<query> queries;

  query q(table);

  //First query
  //First dim's range and weight
  q.attr(0, 3, 4, 0.3);
  //Second dim's range and weight
  q.attr(0, 2, 4, 0.7);
  q.topk(2);
  queries.push_back(q);

  //Second query
  //First dim's range and weight
  q.attr(0, 1, 3, 0.3);
  //Second dim's range and weight
  q.attr(0, 2, 3, 0.7);
  q.topk(1);
  queries.push_back(q);

  table.build();

  device_vector<int> d_top_indexes;

  //You may directly get the top k indexes if you don't care the matching results.
  topk(table, queries, d_top_indexes);

  //Or you can get matching result first then call the top k.
  device_vector<int> d_c;
  device_vector<float> d_a;
  device_vector<int> d_h;
  int hash_table_size;
  match(table, queries, d_c, d_a, d_h, hash_table_size);
  
  //TODO: Modify topk.
  //topk(d_a, queries, d_top_indexes);

  //If you want to have different top k values with the queries, you can pass them in via a device_vector.
  match(table, queries, d_c, d_a);
  //2 Parts
  device_vector<int> d_tops(2);
  //Top 2 values in the first part;
  d_tops[0] = 2;
  //Top 2 values in the second value;
  d_tops[1] = 2;
  
  //TODO: Modify topk.
  //topk(d_a, d_tops, d_top_indexes);

  //The indexes are stored in the d_top_indexes vector.
  //You can transfer it back to host_vector and using it to select the
  //instances in the raw_data.
  //int i, index;
  //vector<string>& row;
  //for (i = 0; i < 4; i++)
  //  {
  //    index = d_top_indexes[i];
  //    //The top matched row.
  //    row = *data.row(index);
  //  }

  printf(">>>>>>>>>>>>>Successful topk searching, the searching result is stored in d_top_indexes vector;\n");
  return 0;
}
