#include "../src/GaLG.h" //for ide: change from <GaLG.h> to "../src/GaLG.h"
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
#include <thrust/system_error.h>
#include <queue>

#define DEFAULT_TOP_K 5

using namespace GaLG;
using namespace std;

u64 MAX_ITEM_NUM = 0;;

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

void read_query(inv_table& table,
		        const char* fname,
		        vector<query>& queries,
		        int num_of_queries,
		        int num_of_query_dims,
		        int radius,
		        int topk,
		        float selectivity)
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
			query q(table, j);
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
			if(selectivity > 0.0f)
			{
				q.selectivity(selectivity);
				q.apply_adaptive_query_range();
			}
			queries.push_back(q);
			++j;
		}
	}

	ifile.close();

	printf("Finish reading queries! %d queries are loaded.\n", num_of_queries);
}

void match_test(inv_table& table,
				const char * dfname,
				int num_of_queries,
				int num_of_query_dims,
				int radius,
				float hash_table_size_,
				int bitmap_bits,
				int num_of_query_print,
				int num_of_hot_dims,
				int hot_dim_threshold,
				float selectivity) throw()
{
	  int device_count;
	  cudaGetDeviceCount(&device_count);
	  cudaSetDevice(device_count - 1);
	  cudaDeviceReset();

	  u64 timestart, timestop, totalstart;

	  vector<query> queries;

	  totalstart = timestart = getTime();
	  printf("Start creating query...\n");

	  //printf("filename is %s.\n", dfname);
	  read_query(table, dfname, queries, num_of_queries, num_of_query_dims, radius, DEFAULT_TOP_K, selectivity);

	  timestop = getTime();
	  printf("Finish creating %d query. Time elapsed: %f ms. \n", queries.size(), getInterval(timestart, timestop));

	  device_vector<data_t> d_data;
	  int hash_table_size = hash_table_size_ * table.i_size() + 1;
	  //match(table, queries, d_data, hash_table_size, bitmap_bits, num_of_hot_dims, hot_dim_threshold);// for ide:
	  if(GALG_ERROR){
		  GALG_ERROR = false;
		  cudaDeviceReset();
		  return;
	  }
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
		  if(non_zero_count > MAX_ITEM_NUM){
			  MAX_ITEM_NUM = non_zero_count;
		  }

	  }
	  printf("[Info] Average Non-zero Item: %f\n", total_non_zero_count / (float)queries.size());

	  if(num_of_query_print){
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
	  }


//	  for(int i = 0; i < queries.size() && i < num_of_query_print;++i){
//		  printf("Matching Result for query %d:\n", i);
//		  for(j = 0; j < hash_table_size; ++j){
//			   hd = h_data[hash_table_size * i + j];
//			   printf("%d %.5f %u\n", j, hd.aggregation, hd.id);
//			  //printf("%3d. Count: %3u, Aggregation: %5.2f, Index: %3u\n", j, hd.count, hd.aggregation, hd.id);
//		  }
//		  printf("-------\n");
//	  }

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
				const int top_k_size,
				const int num_of_hot_dims,
				const int hot_dim_threshold,
				const float selectivity)throw()
{
	 cudaDeviceReset();
	 int device_count;
	 cudaGetDeviceCount(&device_count);
	 cudaSetDevice(device_count - 1);

	  u64 timestart, timestop, totalstart;

	  vector<query> queries;

	  totalstart = timestart = getTime();
	  printf("Start creating query...\n");

	  read_query(table, dfname, queries, num_of_queries, num_of_query_dims, radius, top_k_size, selectivity);

	  timestop = getTime();
	  printf("Finish creating query. Time elapsed: %f ms. \n", getInterval(timestart, timestop));


	  device_vector<int> d_topk;
	  int hash_table_size = hash_table_size_ * table.i_size() + 1;
	  printf("hash table size: %d\n", hash_table_size);

	  timestart = getTime();
	  //GaLG::topk(table, queries, d_topk, hash_table_size, bitmap_bits, num_of_query_dims, num_of_hot_dims, hot_dim_threshold);//for ide:
	  if(GALG_ERROR){
		  cudaDeviceReset();
		  return;
	  }
	  timestop = getTime();
	  GALG_TIME += (timestop - timestart);
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

	  u32 max_topk = 0, min_topk = 0 - 1, sum_topk = 0;
	  typedef struct _topk_pair{
		  u32 index;
		  u32 sum;
	  }topk_pair;
	  topk_pair null_pair;
	  null_pair.index = -1;
	  null_pair.sum = -1;
	  topk_pair pairs[3] = {null_pair, null_pair, null_pair};


	  for(int i = 0; i < queries.size(); ++i)
	  {
		  int q_size = queries[i].topk();
		  int temp_sum = 0;
		  for(int j =0; j < q_size; ++j)
		  {
			  bool topk_fault = 1;
			  if(h_topk[q_size * i + j] % 256 == 0)
			  {
				  for(int z = j + 1; z < q_size; ++z)
				  {
					  if(h_topk[q_size * i + z] % 256 != 0)
					  {
						  topk_fault = 0;
						  break;
					  }

				  }
			  } else {
				  topk_fault = 0;
			  }
			  if(topk_fault)
			  {
				  break;
			  } else {
				  temp_sum ++;
			  }

		  }
		  if(temp_sum < min_topk) min_topk = temp_sum;
		  if(temp_sum > max_topk) max_topk = temp_sum;
		  sum_topk += temp_sum;

		  if(temp_sum < pairs[0].sum)
		  {
			  topk_pair temp_pair;
			  temp_pair.sum = temp_sum;
			  temp_pair.index = i;
			  pairs[0] = temp_pair;
		  }

		  for(int j = 0; j < 2; ++j)
		  {
			  for(int z = j+1; z < 3; ++z)
			  {
				  if(pairs[j].sum < pairs[z].sum)
				  {
					  topk_pair temp_pair;
					  temp_pair.sum = pairs[z].sum;
					  temp_pair.index = pairs[z].index;
					  pairs[z] = pairs[j];
					  pairs[j] = temp_pair;
				  }
			  }

		  }
	  }
	  printf("[Info] Topk num of results: max %d, min %d, avg %.2f.\n", max_topk, min_topk, sum_topk/(double)queries.size());
	  printf("[Info] Min topk: %d: %d, %d: %d, %d:%d.\n", pairs[0].index, pairs[0].sum,pairs[1].index, pairs[1].sum,pairs[2].index, pairs[2].sum);
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
	  parser::csv(dfname, data, false);
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

//int main(int argc, char * argv[])
//{
//	std::vector<query> queries;
//	inv_table table;
//	build_table(table, "/media/hd1/home/luanwenhao/TestData2Wenhao/random/tst_d128_n100k.csv", 128);
//	read_query(table, "/media/hd1/home/luanwenhao/TestData2Wenhao/random/tst_d128_n100k.csv", queries, 100, 128, 0, 100);
//
//	GaLG::GaLG_Config config;
//	config.count_threshold = 4;
//	config.dim = 128;
//	config.hashtable_size = 0.2;
//	config.num_of_topk = 100;
//	config.query_radius = 0;
//	config.use_device = 1;
//
//	std::vector<int> result;
//
//	printf("Launching knn functions...\n");
//	GaLG::knn_search(table, queries, result, config);
//
//	for(int i = 0; i < 10; ++i)
//	{
//		printf("Query %d result is: \n\t", i);
//		for (int j = 0; j < 10; ++j)
//		{
//			printf("%d, ", result[i * 100 + j]);
//		}
//		printf("\n");
//	}
//}

int
main6(int argc, char * argv[])//for ide
{

	if(argc == 1)
	{
		printf("Instruction on using this ugly testing function:\n"
			   "  ./this_programme [<option> <option_value>] ...\n"
			   "Options:\n"
			   "-f    Full path to the data csv file. The file will be loaded fully.\n"
			   "-qf   Full path to the query csv file. May not be loaded fully.\n"
			   "        See [-q] option.\n"
			   "-q    Number of queries to be loaded from the query file.\n"
			   "-d    Number of dimensions of the data and queries.\n"
			   "-t    Number of topk's for queries.\n"
			   "-r    Radius of the to be read query file. It will extend the scanned\n"
			   "        buckets to value + radius and value - radius.\n"
			   "-s    The query selectivity. If set > 0, it will extend the scanned\n"
			   "        buckets to cover enough data to match the selectivity.\n"
			   "        Note that if radius is too large, the selectivity setting will\n"
			   "        not shrink the query range.\n"
			   "-b    The bitmap threshold. Number of bits to be used is automatically\n"
			   "        controlled. Here users only need to set the cut-off threshold.\n"
			   "-h    The hash table size in 'hash table / data size' ratio. Set to 1 \n"
			   "        is safe but fewer queries can be processed. Smaller size must be\n"
			   "        used together with large filter thresholds.\n"
			   "-nhd  Number of hot dimensions. If nhd != 0, the programme will do a two-\n"
			   "        stage scan, with first stage scanning non-hot dim and second stage\n"
			   "        scanning hot dim.\n"
			   "-hdt  Hot dimension threshold. If counts in bitmap + number of hot dim > \n"
			   "        hot dimension threshold, it will proceed to count the data points,\n"
			   "        otherwise skipped (we are confident that it will not be a topk).\n"
			   "-p    Number of queries' topk results to be printed.\n"
			   "-n    Number of tests to be run. Average running time will be printed.\n");
		return 0;
	}
  std::string fname,qfname, lastfname;

  int num_of_query = 1,
	  num_of_dim = -1,
	  radius = 0,
	  num_of_query_printing = 0,
	  num_of_topk =5,
	  bitmap_threshold = 2,
	  num_of_tests = 1,
	  num_of_hot_dims = 0,
	  hot_dim_safe_threshold = bitmap_threshold;
  float selectivity = -1.0f;
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
			  printf("Using default/last hashtable ratio: %f.\n", hashtable);
			}

			if(cmd_option_exists(s, e, "-b"))
			{
				bitmap_threshold = stoi(get_cmd_option(s, e, "-b"));
			} else {
			  printf("Using default/last bitmap threshold: %d.\n", bitmap_threshold);
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
		  	  printf("Using default/last number of topk items: %d.\n", num_of_topk);
		    }
		    if(cmd_option_exists(s, e, "-n"))
		    {
		  	  num_of_tests = stoi(get_cmd_option(s, e, "-n"));
		    } else {
		  	  printf("Using default/last number of tests: %d.\n", num_of_tests);
		    }
		    if(cmd_option_exists(s, e, "-nhd"))
		    {
		    	num_of_hot_dims = stoi(get_cmd_option(s, e, "-nhd"));
		    } else {
		    	printf("Using default/last number of hot dimensions: %d.\n", num_of_hot_dims);
		    }
		    if(cmd_option_exists(s, e, "-hdt"))
		    {
		    	hot_dim_safe_threshold = stoi(get_cmd_option(s, e, "-hdt"));
		    } else {
		    	printf("Using default/last hot dim safe threshold: %d.\n", hot_dim_safe_threshold);
		    }
		    if(cmd_option_exists(s, e, "-s"))
		    {
		    	selectivity = stof(get_cmd_option(s,e,"-s"));
		    } else {
		    	printf("Using default/last selectivity: %f.\n", selectivity);
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
	    try{
	    	GALG_TIME = 0ull;
	    	GALG_ERROR= false;
		    if(function == 0 && !error)
		    {
		    	MAX_ITEM_NUM = 0ull;
		  	  match_test(table,
		  			     qfname.c_str(),
		  			     num_of_query,
		  			     num_of_dim, radius,
		  			     hashtable,
		  			     bitmap_threshold,
		  			     num_of_query_printing,
		  			     num_of_hot_dims,
		  			     hot_dim_safe_threshold,
		  			     selectivity);
		  	  printf("Max number of items in query hashtables: %llu.\n", MAX_ITEM_NUM);
		    }
		    else if(function == 1 && !error)
		    {

		      for(int i = 0; i < num_of_tests && !GALG_ERROR; ++i){
		    	  topk_test(table,
		    			    qfname.c_str(),
		    			    num_of_query,
		    			    num_of_dim,
		    			    radius,
		    			    hashtable,
		    			    bitmap_threshold,
		    			    num_of_query_printing,
		    			    num_of_topk,
		    			    num_of_hot_dims,
		    			    hot_dim_safe_threshold,
		    			    selectivity);
		      }
		      if(num_of_tests != 1 && !GALG_ERROR)
		      {
		    	  printf("Average topk time is %f for %d tests.\n", GALG_TIME / (double)(num_of_tests*1000), num_of_tests);
		      }


		    }
		    else
		    {
		  	  printf("Shutting down kernel... Please try again.\n");
		    }
	    }
	    catch(thrust::system_error  & e)
	    {
	    	printf("%s\n", e.what());
	    	cudaDeviceReset();
	    	printf("Please try again.\n");
	    }
	    catch(std::bad_alloc  & e)
	    {
	    	printf("%s\n", e.what());
	    	cudaDeviceReset();
	    	printf("Please try again.\n");
	    }
	    catch(MemException  & e)
	    {
	    	printf("%s\n", e.what());
	    	cudaDeviceReset();
	    	printf("Please try again.\n");
	    }
	    catch(...)
	    {
	    	printf("Unkown error!\n");
	    	printf("Please try again.\n");
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
