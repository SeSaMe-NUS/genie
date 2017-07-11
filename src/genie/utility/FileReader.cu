/*! \file FileReader.cc
 *  \brief Implementation of functions in FileReader.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <map>
#include <iostream>

#include <genie/utility/Logger.h>
#include <genie/query/query.h>
#include <genie/table/inv_table.h>

#include "FileReader.h"

using namespace std;
using namespace genie::query;
using namespace genie::table;


vector<string> split(string& str, const char* c)
{
	char *cstr, *p;
	vector<string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL)
	{
		res.push_back(p);
		p = strtok(NULL, c);
	}
	delete[] cstr;
	return res;
}

string eraseSpace(string origin)
{
	int start = 0;
	while (origin[start] == ' ')
		start++;
	int end = origin.length() - 1;
	while (origin[end] == ' ')
		end--;
	return origin.substr(start, end - start + 1);
}


void genie::utility::read_file(vector<vector<int> >& dest, const char* fname, int num)
{
	string line;
	ifstream ifile(fname);

	dest.clear();

	if (ifile.is_open())
	{
		int count = 0;
		while (getline(ifile, line) && (count < num || num < 0))
		{
			vector<int> row;
			vector<string> nstring = split(line, ", ");
			unsigned int i;
			for (i = 0; i < nstring.size(); ++i)
			{
				int int_value = atoi(eraseSpace(nstring[i]).c_str());
				row.push_back(int_value);
			}
			dest.push_back(row);
			count++;
		}
		Logger::log(Logger::INFO, "Finish reading file!");
		Logger::log(Logger::DEBUG, "%d rows are read into memory!",
				dest.size());
	}

	ifile.close();
}

//Read new format query data
//Sample data format
//qid dim value selectivity weight
// 0   0   15     0.04        1
// 0   1   6      0.04        1
// ....
void genie::utility::read_query(std::vector<genie::utility::attr_t>& data, const char* file_name,
		int num)
{

	string line;
	ifstream ifile(file_name);

	data.clear();
	int count = num;
	int total = 0;
	attr_t attr;
	if (ifile.is_open())
	{

		while (getline(ifile, line) && count != 0)
		{

			vector<string> nstring = split(line, ", ");

			if (nstring.size() == GPUGENIE_QUERY_NUM_OF_FIELDS)
			{
				count--;
				total++;
				attr.qid = atoi(nstring[GPUGENIE_QUERY_QID_INDEX].c_str());
				attr.dim = atoi(nstring[GPUGENIE_QUERY_DIM_INDEX].c_str());
				attr.value = atoi(nstring[GPUGENIE_QUERY_VALUE_INDEX].c_str());
				attr.sel = atof(
						nstring[GPUGENIE_QUERY_SELECTIVITY_INDEX].c_str());
				attr.weight = atof(
						nstring[GPUGENIE_QUERY_WEIGHT_INDEX].c_str());
				data.push_back(attr);
			}
		}
	}

	ifile.close();

	Logger::log(Logger::INFO, "Finish reading query data!");
	Logger::log(Logger::DEBUG, "%d attributes are loaded.", total);
}

//Read old format query data: same format as data files
void genie::utility::read_query(genie::table::inv_table& table, const char* fname,
		vector<genie::query::Query>& queries, int num_of_queries, int num_of_query_dims,
		int radius, int topk, float selectivity)
{

	string line;
	ifstream ifile(fname);

	queries.clear();
	queries.reserve(num_of_queries);

	if (ifile.is_open())
	{
		int j = 0;
		while (getline(ifile, line) && j < num_of_queries)
		{

			vector<string> nstring = split(line, ", ");
			unsigned int i;
			Query q(table, j);
			for (i = 0; i < nstring.size() && i < (unsigned int) num_of_query_dims; ++i)
			{
				string myString = eraseSpace(nstring[i]);
				int value = atoi(myString.c_str());

				q.attr(j, value - radius < 0 ? 0 : value - radius,
						value + radius, 1, i);
			}
			q.topk(topk);
			if (selectivity > 0.0f)
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
	Logger::log(Logger::DEBUG, "%d queries are loaded.", num_of_queries);
}

