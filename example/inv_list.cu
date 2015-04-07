#include <GaLG.h>
#include <vector>
#include <stdio.h>
#include <string>

using namespace GaLG;
using namespace std;

struct mm {
  int min;
  int max;
};

int
converter(string& value, void* d)
{
  int num = atoi(value.c_str());
  mm* minmax_value = (mm*) d;

  if(num < (minmax_value->min))
    return minmax_value ->min;
  if(num > (minmax_value->max))
    return minmax_value ->max;
  return num;
}

int
main()
{
  //Get datas from a csv.
  raw_data data;
  parser::csv("../static/t1.csv", data);

  //Get column.
  vector<string>& col = *data.col(0); //The first column;

  inv_list list;

  //The inverted index list of the first column.
  list.invert(col);

  //The inverted index list of the column g2.
  list.invert(data.col("g2"));

  //Invert a string vector using a converter.
  mm minmax_value;
  minmax_value.min = 0;
  minmax_value.max = 100;
  list.invert(data.col("g2"), &converter, &minmax_value);

  //indexes of value 2
  vector<int>& indexes = *list.index(2);

  //min max value
  int min = list.min();
  int max = list.max();

 printf(">>>>>>>>>>>>>Successful reading, the minimum =%d the maximum =%d;\n",min,max);
  return 0;
}
