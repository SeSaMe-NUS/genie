#ifndef GaLG_inv_list_h
#define GaLG_inv_list_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  class inv_list {
  private:
    int _size;
    pair<int, int> _bound;
    vector<vector<int> > _inv;

  public:
    inv_list(){}
    inv_list(vector<int>&);
    inv_list(vector<int>*);
    inv_list(vector<string>&);
    inv_list(vector<string>*);

    int min();
    int max();
    int size();
    void invert(vector<int>&);
    void invert(vector<int>*);
    void invert(vector<string>&);
    void invert(vector<string>*);
    void invert(vector<string>&, int(*stoi)(string&, void*), void*);
    void invert(vector<string>*, int(*stoi)(string&, void*), void*);
    bool contains(int);
    vector<int>* index(int);
  };
}

#endif