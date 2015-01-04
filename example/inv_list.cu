#include <GaLG.h>
#include <vector>
#include <string>

using namespace GaLG;
using namespace std;

int
main()
{
  //Get datas from a csv.
  raw_data data;
  parser::csv("static/t1.csv", data);

  //Get column.
  vector<string>& col = *data.col(0); //The first column;

  inv_list list;

  //The inverted index list of the first column.
  list.invert(col);

  //The inverted index list of the column g2.
  list.invert(data.col("g2"));

  //indexes of value 2
  vector<int>& indexes = *list.index(2);

  //min max value
  int min = list.min();
  int max = list.max();
  return 0;
}
