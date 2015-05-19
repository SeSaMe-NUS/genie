#include "GaLG.h"
#include <stdio.h>
#include <string>

using namespace GaLG;
using namespace std;

int
main()
{
  raw_data data;
  parser::csv("../static/t1.csv", data);

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

  device_vector<data_t> d_data;
  int hash_table_size;
  match(table, queries, d_data, hash_table_size);

  std::vector<data_t> h_data(hash_table_size * queries.size());
  thrust::copy(d_data.begin(), d_data.begin()+hash_table_size * queries.size(), h_data.begin());
  cudaDeviceSynchronize();

  printf("hash table size: %d\n", hash_table_size);

  int i,j;
  data_t hd;
  for(i = 0; i < queries.size();++i){
	  printf("Matching Result for query %d:\n", i);
	  for(j = 0; j < hash_table_size; ++j){
		   hd = h_data[hash_table_size * i + j];
		  printf("%3d. Count: %3u, Aggregation: %3.2f, Index: %3u\n", j, hd.count, hd.aggregation, hd.id);
	  }
	  printf("-------\n");
  }
  printf(">>>>>>>>>>>>>Successful maching, the matching result is stored in d_data;\n");
  return 0;
}
