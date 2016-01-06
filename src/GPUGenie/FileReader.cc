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
#include <stdexcept>
#include <map>
#include <iostream>

#include "Logger.h"

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
		Logger::log(Logger::INFO, "Finish reading file!");
		Logger::log(Logger::DEBUG, "%d rows are read into memory!", dest.size());
	}

	ifile.close();
}

//Read new format query data
//Sample data format
//qid dim value selectivity weight
// 0   0   15     0.04        1
// 0   1   6      0.04        1
// ....
void GPUGenie::read_query(std::vector<attr_t>& data, const char* file_name, int num)
{

	string line;
	ifstream ifile(file_name);

	data.clear();
	int count = num;
	int total = 0;
	attr_t attr;
	if (ifile.is_open()) {

		while (getline(ifile, line) && count != 0) {

			vector<string> nstring = split(line, ", ");

			if(nstring.size() == GPUGENIE_QUERY_NUM_OF_FIELDS){
				count --;
				total ++;
				attr.qid = atoi(nstring[GPUGENIE_QUERY_QID_INDEX].c_str());
				attr.dim = atoi(nstring[GPUGENIE_QUERY_DIM_INDEX].c_str());
				attr.value = atoi(nstring[GPUGENIE_QUERY_VALUE_INDEX].c_str());
				attr.sel = atof(nstring[GPUGENIE_QUERY_SELECTIVITY_INDEX].c_str());
				attr.weight = atof(nstring[GPUGENIE_QUERY_WEIGHT_INDEX].c_str());
				data.push_back(attr);
			}
		}
	}

	ifile.close();

	Logger::log(Logger::INFO, "Finish reading query data!");
	Logger::log(Logger::DEBUG,"%d attributes are loaded.", total);
}

//Read old format query data: same format as data files
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

	Logger::log(Logger::INFO, "Finish reading queries!");
	Logger::log(Logger::DEBUG,"%d queries are loaded.", num_of_queries);
}

void
GPUGenie::csv2binary(const char* csvfile, const char* binaryfile, bool app_write)
{
     vector<vector<int> > data;
     GPUGenie::read_file(data, csvfile, -1);
     if(data.size()<=0||data[0].size()<=0)
     {
    	  Logger::log(Logger::ALERT,"In processsing csv2binary: cannot read data from csv:%s ",csvfile);
          return;
     }
     unsigned int row_num = data.size();
     unsigned int item_num =0 ;
     unsigned int *index;
     int *_data;

     index = (unsigned int*)malloc(row_num*sizeof(unsigned int));

     index[0] = 0;
     for(unsigned int i=0 ; i<data.size()-1 ; ++i)
     {
         index[i+1] = index[i] + data[i].size();
	          item_num += data[i].size();
     }
     item_num += data[data.size()-1].size();
    _data = (int*)malloc(sizeof(int)*item_num);
    unsigned int k=0;
    for(unsigned int i=0; i<data.size(); ++i)
    {
		for(unsigned int j=0; j<data[i].size(); ++j)
		{
			_data[k] = data[i][j];
			k++;
		}
    } 

    Logger::log(Logger::DEBUG, "Written item_num:%d", item_num);
    Logger::log(Logger::DEBUG, "Written row_num: ", row_num);

    ofstream ofs(binaryfile, ios::trunc | ios::binary);
    if(app_write){
        ofs.close();
        ofstream ofs(binaryfile, ios::app | ios::binary );
    }

    if(!ofs.is_open())
    {
    	 Logger::log(Logger::ALERT,"In processing csv2binary: cannot open binary file: %s",binaryfile);
         return;
    }

    ofs.write((char*)&item_num, sizeof(unsigned int));
    ofs.write((char*)&row_num, sizeof(unsigned int));
    ofs.write((char*)_data, sizeof(int)*item_num);
    ofs.write((char*)index, sizeof(unsigned int)*row_num);

    ofs.close();

    free(_data);
    free(index);

    return;
}


void
GPUGenie::read_file(const char* fname, int **_data, unsigned int& item_num,
                    unsigned int **_index, unsigned int& row_num)
{
     ifstream ifs(fname, ios::binary);
     if(!ifs.is_open())
     {
    	 Logger::log(Logger::ALERT,"In read_file from binary: cannot open file: %s", fname);
         return;
     }
     ifs.read((char*)&item_num, sizeof(unsigned int));
     ifs.read((char*)&row_num, sizeof(unsigned int));
     *_data = (int*)malloc(sizeof(int)*item_num);
     *_index = (unsigned int*)malloc(sizeof(unsigned int)*row_num);
     ifs.read((char*)*_data, sizeof(int)*item_num);
     ifs.read((char*)*_index, sizeof(unsigned int)*row_num);
     ifs.close();
}
















/* namespace GPUGenie */
