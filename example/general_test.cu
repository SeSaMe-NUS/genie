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
#include <algorithm>


#define DEFAULT_TOP_K 5

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

void read_query(inv_table& table, const char* fname, vector<query>& queries, int num_of_queries, int num_of_query_dims, int radius, int topk)
{

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
			q.topk(topk);
			queries.push_back(q);
			++j;
		}
	}

	ifile.close();
}

void match_test(inv_table& table,
				const char * dfname,
				int num_of_queries,
				int num_of_query_dims,
				int radius,
				float hash_table_size_,
				int bitmap_bits,
				int num_of_query_print)
{
	 cudaDeviceReset();
	 int device_count;
	 cudaGetDeviceCount(&device_count);
	 cudaSetDevice(device_count - 1);

	  u64 timestart, timestop, totalstart;

	  vector<query> queries;

	  totalstart = timestart = getTime();
	  printf("Start creating query...\n");

	  query q(table);
	  //printf("filename is %s.\n", dfname);
	  read_query(table, dfname, queries, num_of_queries, num_of_query_dims, radius, DEFAULT_TOP_K);

	  timestop = getTime();
	  printf("Finish creating %d query. Time elapsed: %f ms. \n", queries.size(), getInterval(timestart, timestop));

	  device_vector<data_t> d_data;
	  int hash_table_size = hash_table_size_ * table.i_size() + 1;
	  match(table, queries, d_data, hash_table_size, bitmap_bits);

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

	  for(int i = 0; i < queries.size(); ++i)
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
		  //printf("\t %u: %.5f%%\n", it->first, 100 * (it->second / (double)all_count));
		  printf("\t %u: %u\n", it->first, it->second);
	  }

	  for(int i = 0; i < queries.size() && i < num_of_query_print;++i){
		  printf("Matching Result for query %d:\n", i);
		  for(j = 0; j < hash_table_size; ++j){
			   hd = h_data[hash_table_size * i + j];
			   printf("%d %.5f %u\n", j, hd.aggregation, hd.id);
			  //printf("%3d. Count: %3u, Aggregation: %5.2f, Index: %3u\n", j, hd.count, hd.aggregation, hd.id);
		  }
		  printf("-------\n");
	  }

	  printf(">>>>>>>>>>>>>Successful matching, the matching result is stored in d_data;\n");
	  timestop = getTime();
	  printf("Finish testing. Time elapsed: %f ms. \n", getInterval(totalstart, timestop));
}

void topk_test( inv_table& table,
				const char * dfname,
				const int num_of_queries,
				const int num_of_query_dims,
				const int radius,
				const float hash_table_size_,
				const int bitmap_bits,
				const int num_of_query_print,
				const int top_k_size)
{
	 cudaDeviceReset();
	 int device_count;
	 cudaGetDeviceCount(&device_count);
	 cudaSetDevice(device_count - 1);

	  u64 timestart, timestop, totalstart;

	  vector<query> queries;

	  totalstart = timestart = getTime();
	  printf("Start creating query...\n");

	  read_query(table, dfname, queries, num_of_queries, num_of_query_dims, radius, top_k_size);

	  timestop = getTime();
	  printf("Finish creating query. Time elapsed: %f ms. \n", getInterval(timestart, timestop));


	  device_vector<int> d_topk;
	  int hash_table_size = hash_table_size_ * table.i_size() + 1;
	  printf("hash table size: %d\n", hash_table_size);

	  timestart = getTime();
	  GaLG::topk(table, queries, d_topk, hash_table_size, bitmap_bits);
	  timestop = getTime();
	  printf("Topk takes %f ms.\n", getInterval(timestart, timestop));

	  printf("Starting copying device result to host...\n");
	  timestart = getTime();
	  host_vector<int> h_topk(d_topk);
	  cudaDeviceSynchronize();
	  timestop = getTime();
	  printf("Finish copying result. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  for(int i = 0; i < queries.size() && i < num_of_query_print; ++i)
	  {
		  printf("The top %d of query %d is \n", queries[i].topk(),i);
		  if(queries[i].topk() > 0) printf("%d", h_topk[0 + i * queries[i].topk()]);
		  for(int j = 1; j < queries[i].topk(); ++j)
		  {
			  printf(", %d", h_topk[i * queries[i].topk() + j]);
		  }
		  printf("\n");
	  }

	  printf(">>>>>>>>>>>>>Successful topk searching.\n");
	  timestop = getTime();
	  printf("Finish testing. Time elapsed: %f ms. \n", getInterval(totalstart, timestop));
}

void
build_table(inv_table& table,const char * dfname,const int num_of_dims)
{
	  u64 timestart, timestop;
	  raw_data data;

	  printf("Start loading data...\n");
	  timestart = getTime();
	  parser::csv(dfname, data);
	  timestop = getTime();
	  printf("Finish loading data. Time elapsed: %f ms. \n", getInterval(timestart, timestop));

	  inv_list list;

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
	  printf("Start building table...\n");
	  timestart = getTime();
	  table.build();
	  timestop = getTime();
	  printf("Finish building table. Time elapsed: %f ms. \n", getInterval(timestart, timestop));
}

std::string get_cmd_option(std::vector<std::string>::iterator& begin, std::vector<std::string>::iterator& end, const std::string & option)
{
	std::vector<std::string>::iterator itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmd_option_exists(std::vector<std::string>::iterator& begin, std::vector<std::string>::iterator& end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int stoi(std::string str)
{
	int result = atoi(str.c_str());
	if(str.empty() ||(eraseSpace(str) != std::string("0") && result == 0)){
		throw 0;
	}
	return result;
}

float stof(std::string str)
{
	float result = atof(str.c_str());
	if(str.empty() ||  ( eraseSpace(str) != std::string("0") && result == 0.0f)){
		throw 0;
	}
	return result;
}

int
main(int argc, char * argv[])
{

  std::string fname,qfname, lastfname;

  int num_of_query = 1, num_of_dim = -1, radius = 0, num_of_query_printing = 0, num_of_topk =5, bitmap_bits = 2;
  float hashtable = 1.0f;
  std::vector<std::string> ss;
  inv_table table;
  int function = -1;

  for(int i = 1;i < argc; ++i)
  {
	  ss.push_back(std::string(argv[i]));
  }

  while(1)
  {
	    bool error = false;
	    std::vector<std::string>::iterator s = ss.begin();
	    std::vector<std::string>::iterator e = ss.end();

	    try{
			if(cmd_option_exists(s, e, "-f"))
			{
			  lastfname = fname;
			  fname =  get_cmd_option(s, e, "-f");
			} else {
			  if(!fname.empty())
			  {
				  printf("Using last file: %s.\n", fname.c_str());
			  }
			  else
			  {
				  printf("Please indicate data file path using -f.\n");
				  error =true;
			  }
			}

			if(cmd_option_exists(s, e, "-qf"))
			{
				qfname = get_cmd_option(s, e, "-qf");
			} else {
				if(qfname.empty()) qfname = fname;
			}

			if(cmd_option_exists(s, e, "-q"))
			{
			  num_of_query = stoi(get_cmd_option(s, e, "-q"));
			} else {
			  printf("Using default/last number of query: %d.\n", num_of_query);
			}

			if(cmd_option_exists(s, e, "-d"))
			{
			  num_of_dim = stoi(get_cmd_option(s, e, "-d"));
			} else {
			  if(num_of_dim != -1)
			  {
				  printf("Using last number of dim: %d.\n", num_of_dim);
			  }
			  else
			  {
				  printf("Please indicate data dimension using -d.\n");
				  error =true;
			  }

			}

			if(cmd_option_exists(s, e, "-r"))
			{
			  radius = stoi(get_cmd_option(s, e, "-r"));
			} else {
			  printf("Using default/last radius: %d.\n", radius);
			}

			if(cmd_option_exists(s, e, "-h"))
			{
			  hashtable = stof(get_cmd_option(s, e, "-h"));
			} else {
			  printf("Using default/last hashtable ratio: %.1f.\n", hashtable);
			}

			if(cmd_option_exists(s, e, "-b"))
			{
			  bitmap_bits = stoi(get_cmd_option(s, e, "-b"));
			} else {
			  printf("Using default/last bitmap bits: %d.\n", bitmap_bits);
			}

			if(cmd_option_exists(s, e, "-p"))
			{
			  num_of_query_printing = stoi(get_cmd_option(s, e, "-p"));
			} else {
			  printf("Using default/last number of query to be printed: %d.\n", num_of_query_printing);
			}

		    if(cmd_option_exists(s, e, "-t"))
		    {
		  	  num_of_topk = stoi(get_cmd_option(s, e, "-t"));
		    } else {
		  	  printf("Using default number of topk items: %d.\n", num_of_topk);
		    }
	    } catch(exception& e){
	    	printf("Something wrong with your parameter: %s.\n", e.what());
	    	error = true;
	    }
	    if(!error && (cmd_option_exists(s, e, "match") || cmd_option_exists(s, e, "topk")))
	    {
	  	  if(lastfname != fname)
	  	  {
	  		  build_table(table, fname.c_str(), num_of_dim);
	  	  }
	    }

	    if(cmd_option_exists(s, e, "match"))
	    {
	  	  function = 0;
	    }
	    else if(cmd_option_exists(s, e, "topk"))
	    {
	  	  function = 1;
	    }
	    else if(function == -1)
	    {
	  	  error = true;
	  	  printf("Please specify function.\n");
	    }
	    else
	    {
	  	  printf("Using last function - %s.\n", function == 0? "match" : "topk");
	    }

	    if(function == 0 && !error)
	    {
	  	  match_test(table, qfname.c_str(), num_of_query, num_of_dim, radius, hashtable, bitmap_bits, num_of_query_printing);
	    }
	    else if(function == 1 && !error)
	    {
	  	  topk_test(table, qfname.c_str(), num_of_query, num_of_dim, radius, hashtable, bitmap_bits,num_of_query_printing, num_of_topk);

	    }
	    else
	    {
	  	  printf("Shutting down kernel... Please try again.\n");
	    }

	    printf("[Ctrl + D] to exit, [Enter] to run with same config, or change config to run again.\n");
	    char choice = (char) getchar();
	    if(EOF == choice)
	    {
	  	  return 0;
	    }
	    else if( '\n' != choice)
	    {
	  	  char temp[1000];
	  	  scanf("%[^\n]", temp);
	  	  std::string st(temp);
	  	  st.insert(0, 1, choice);
	  	  ss = split(st, " ");
	  	  getchar();
	    } else {
	  	  ss.clear();
	    }
  }

}
