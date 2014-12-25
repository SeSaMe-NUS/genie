#ifndef GaLG_raw_data_h
#define GaLG_raw_data_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  struct raw_data {
    vector<string> _meta;
    vector<vector<string> > _instance;
    vector<vector<string> > _transpose;

    void clear();
    int m_size();
    int i_size();
    string* meta(int);
    int meta(string);
    vector<string>* row(int);
    vector<string>* col(string);
    vector<string>* col(int);
  };
}

#endif
