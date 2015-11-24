#include <GPUGenie.h> //for ide: change from <GPUGenie.h> to "../src/GPUGenie.h"
#include <stdio.h>
#include <string>

using namespace GPUGenie;
using namespace std;

int
main()//for ide: from main to main7
{
  //The raw_data container.
  raw_data data;

  //Fetch information from a csv file.
  parser::csv("static/t1.csv", data);

  printf("The number of cols: %d\n", data.m_size());
  printf("The number of rows: %d\n", data.i_size());

  vector<string>& row = *data.row(0);
  printf("The first row, first value: %s\n", row[0].c_str());
  printf("The first row, second value: %s\n", row[1].c_str());

  vector<string>& col = *data.col(0);
  printf("The first col, first value: %s\n", col[0].c_str());
  printf("The first col, second value: %s\n", col[1].c_str());

  return 0;
}
