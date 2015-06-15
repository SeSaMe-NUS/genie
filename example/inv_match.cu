#include "GaLG.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <ctime>
#include <map>
#include <bitset>
#include <vector>

using namespace GaLG;
using namespace std;

vector<string> split(string& str, const char* c) {
	char *cstr, *p;
	vector<string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL) {
		res.push_back(p);
		p = strtok(NULL, c);
	}
	delete[] cstr;
	return res;
}

string eraseSpace(string origin) {
	int start = 0;
	while (origin[start] == ' ')
		start++;
	int end = origin.length() - 1;
	while (origin[end] == ' ')
		end--;
	return origin.substr(start, end - start + 1);
}

void read_query(inv_table& table, const char* fname, vector<query>& queries, int num_of_queries, int num_of_query_dims, int radius) {

	string line;
	ifstream ifile(fname);

	queries.clear();
	queries.reserve(num_of_queries);

	if (ifile.is_open()) {
		int j = 0;
		while (getline(ifile, line)&& j < num_of_queries) {

			vector<string> nstring = split(line, ", ");
			int i;
			query q(table);
			for(i = 0; i < nstring.size() && i<num_of_query_dims; ++i){
				string myString = eraseSpace(nstring[i]);
				//cout<<"my string"<<myString<<endl;
				int value = atoi(myString.c_str());

				q.attr(i,
					   value - radius < 0 ? 0 : value - radius,
					   value + radius,
					   1);
			}
			queries.push_back(q);
			++j;
		}
	}

	ifile.close();
}

void test1(void)
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
  int hash_table_size = 0;
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
		  printf("%3d. Aggregation: %5.2f, Index: %3u\n", j, hd.aggregation, hd.id);
	  }
	  printf("-------\n");
  }
  printf(">>>>>>>>>>>>>Successful matching, the matching result is stored in d_data;\n");
}

void test2(const char * dfname, const char * qfname, int num_of_queries, int num_of_dims, int num_of_query_dims, int radius, int hash_table_size_, int num_of_query_print)
{
	  u64 load_elapsed, timestart, timestop, totalstart, total_elapsed;
	  raw_data data;

	  printf("Start loading data...\n");
	  totalstart = timestart = getTime();
	  parser::csv(dfname, data);
	  timestop = getTime();
	  printf("Finish loading data. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  inv_list list;
	  inv_table table;

	  printf("Start inverting data...\n");
	  timestart = getTime();
	  int i;
	  for(i = 0; i < num_of_dims; ++i)
	  {
		  list.invert(data.col(i));
		  table.append(list);
	  }

	  timestop = getTime();
	  printf("Finish inverting data. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  vector<query> queries;

	  timestart = getTime();
	  printf("Start creating query...\n");

	  query q(table);

	  read_query(table, qfname, queries, num_of_queries, num_of_query_dims, radius);

	  timestop = getTime();
	  printf("Finish creating query. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  printf("Start building table...\n");
	  timestart = getTime();
	  table.build();
	  timestop = getTime();
	  printf("Finish building table. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  device_vector<data_t> d_data;
	  int hash_table_size = hash_table_size_;
	  match(table, queries, d_data, hash_table_size);

	  printf("Starting copying device result to host...\n");
	  timestart = getTime();
	  std::vector<data_t> h_data(hash_table_size * queries.size());
	  thrust::copy(d_data.begin(), d_data.begin()+hash_table_size * queries.size(), h_data.begin());
	  cudaDeviceSynchronize();
	  timestop = getTime();
	  printf("Finish copying result. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  printf("hash table size: %d\n", hash_table_size);

	  int j;
	  u64 non_zero_count, total_non_zero_count = 0;
	  data_t hd;
	  std::map<u32, u32> m;

	  std::vector<int> count1(9,0);
	  std::vector<int> count2(9,0);

	  printf("Non-zero Item Number:\n");
	  for(i = 0; i < queries.size(); ++i)
	  {
		  std::map<u32, int> im;
		  non_zero_count = 0;
		  for(j = 0; j < hash_table_size; ++j)
		  {
			  hd = h_data[hash_table_size * i + j];
			  if(hd.aggregation != 0.0f)
			  {
				  ++ non_zero_count;
				  ++ total_non_zero_count;
				  if(m.find(hd.aggregation) == m.end())
				  {
					  m[hd.aggregation] = 0u;
				  }
				  m[hd.aggregation] += 1u;

				  if(im.find(hd.aggregation) == im.end())
				  {
					 im[hd.aggregation] = 0;
				  }
				  im[hd.aggregation] += 1;
			  }
		  }

		  int c1 = int((im[1u]/(float)non_zero_count)*10);
		  count1[c1] ++;
		  int c2 = int((im[2u]/(float)non_zero_count)*10);
		  count2[c2]++;
		  //printf("\t Query %d: %d\n", i, non_zero_count);
	  }
	  printf("[Info] Average Non-zero Item: %f\n", total_non_zero_count / (float)queries.size());
	  printf("Result distribution:\n");
	  u64 all_count = 0ull;
	  for(std::map<u32, u32>::iterator it = m.begin(); it != m.end(); ++it)
	  {
		  all_count += it->second;
	  }
	  for(std::map<u32, u32>::iterator it = m.begin(); it != m.end(); ++it)
	  {
		  printf("\t %u: %.5f%%\n", it->first, 100 * (it->second / (double)all_count));
	  }

	  for(i = 0; i < queries.size() && i < num_of_query_print;++i){
		  printf("Matching Result for query %d:\n", i);
		  for(j = 0; j < hash_table_size; ++j){
			   hd = h_data[hash_table_size * i + j];
			   printf("%d %.5f %u\n", j, hd.aggregation, hd.id);
			  //printf("%3d. Count: %3u, Aggregation: %5.2f, Index: %3u\n", j, hd.count, hd.aggregation, hd.id);
		  }
		  printf("-------\n");
	  }

//	  printf("Count 1 Distribution: \n");
//	  for(i = 0; i < 10; ++i)
//	  {
//		  printf("0.%d-0.%d9 : %d\n",i, i, count1[i]);
//	  }
//	  printf("Count 2 Distribution: \n");
//	  for(i = 0; i < 10; ++i)
//	  {
//		  printf("0.%d-0.%d9 : %d\n",i, i, count2[i]);
//	  }
	  printf(">>>>>>>>>>>>>Successful matching, the matching result is stored in d_data;\n");
	  timestop = getTime();
	  printf("Finish testing. Time elapsed: %f ms. \n", getInterval(totalstart, timestop));
}

int
main(int argc, char * argv[])
{
  if(argc == 2 && argv[1][0] == '1')
  {
	  test1();
  }
  else if((argc == 8 || argc == 9 || argc == 10) && argv[1][0] == '2')
  {
	  if(argc == 8)
		  test2(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), 0, 1);
	  else if (argc == 9)
		  test2(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), 1);
	  else if(argc == 10)
		  test2(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]));
  }
  else
  {
	  printf("Wrong number of arguments provided!\n");
	  printf("Usage:\n"
			 "    To run simple test with 12 data points and 2 queries only:\n"
			 "        inv_match_bin 1 \n"
			 "    To run arbitrary test:\n"
			 "        inv_match_bin 2 <data path> <query path> <num of queries> <num of dims> <num of query dims> <query radius> [<hash table size>] [<num of query result to be printed>]\n"
			 "Please start over again...\n");
  }
  return 0;
}
