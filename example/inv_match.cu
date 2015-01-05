#include <GaLG.h>
#include <stdio.h>
#include <string>

using namespace GaLG;
using namespace std;

int
main()
{
  raw_data data;
  parser::csv("static/t1.csv", data);

  inv_list list;
  inv_table table;

  //Insert first dim
  list.invert(data.col("g1"));
  table.append(list);

  //Insert second dim
  list.invert(data.col("g2"));
  table.append(list);

  vector<query> queries;

  query q(table);

  //First query
  //First dim's range and weight
  q.attr(0, 3, 4, 0.3);
  //Second dim's range and weight
  q.attr(0, 2, 4, 0.7);
  queries.push_back(q);

  //Second query
  //First dim's range and weight
  q.attr(0, 1, 3, 0.3);
  //Second dim's range and weight
  q.attr(0, 2, 3, 0.7);
  queries.push_back(q);

  table.build();

  device_vector<int> d_count;
  device_vector<float> d_aggregation;
  match(table, queries, d_count, d_aggregation);

  //The matching result is stored in d_count and d_aggregation;

  return 0;
}
