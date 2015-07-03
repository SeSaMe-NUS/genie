#include "GaLG.h"

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

using namespace GaLG;
using namespace std;

int main()
{
  srand(time(NULL));
  vector<int> v(100);
  v[0] = rand() % 10 + 1;
  int i;
  for(i=1; i<100; i++)
  {
    v[i] = rand() % 100 + v[0];
  }
  v[70] = rand() % 100 + 1000;

  inv_list inv;
  inv.invert(v);

  assert(inv.min() == v[0]);
  assert(inv.max() == v[70]);
  assert(inv.index(v[0]) != NULL);
  assert((*inv.index(v[0]))[0] == 0);
  assert(inv.index(v[70]) != NULL);
  assert((*inv.index(v[70]))[0] == 70);
  assert(inv.index(10000) == NULL);

  return 0;
}
