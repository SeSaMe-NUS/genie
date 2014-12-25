#ifndef GaLG_inv_list_h
#define GaLG_inv_list_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  class inv_list {
  private:
    pair<int, int> bound;

  public:
    int min();
    int max();
  };
}

#endif