/*
 * FileReader.cpp
 *
 *  Created on: Oct 26, 2015
 *      Author: zhoujingbo
 */

#include "FileReader.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

using namespace GPUGenie;
using namespace std;



namespace GPUGenie {
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
}

void GPUGenie::read_file(vector<vector<int> >& dest,
		        const char* fname,
		        int num)
{
	string line;
	ifstream ifile(fname);

	dest.clear();

	if (ifile.is_open()) {
		int count = 0;
		while (getline(ifile, line) && (count < num || num < 0)) {
			vector<int> row;
			vector<string> nstring = split(line, ", ");
			int i;
			for(i = 0; i < nstring.size(); ++i){
				int int_value = atoi(eraseSpace(nstring[i]).c_str());
				row.push_back(int_value);
			}
			dest.push_back(row);
			count ++;
		}
		printf("%d rows are read into memory!\n", dest.size());
	}

	ifile.close();
}


void GPUGenie::read_query(inv_table& table,
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

/* namespace GPUGenie */
