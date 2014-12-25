#ifndef GaLG_inv_list_h
#define GaLG_inv_list_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  class inv_list {
  private:
    pair<int, int> _bound;
    vector<vector<int> > _inv;

  public:
    int min();
    int max();
    void invert(vector<int>&);
    void invert(vector<string>&);
    void invert(vector<string>&, int(*stoi)(string&, void*), void*);
    bool contains(int);
    vector<int>* index(int);
  };
}

#endif