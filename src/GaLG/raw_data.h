#ifndef GaLG_raw_data_h
#define GaLG_raw_data_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  struct raw_data {
    vector<string> meta;
    vector<vector<string> > instance;

    int m_size();
    int i_size();
    raw_data& select(string, vector<string>&);
    raw_data& select(int, vector<string>&);
  };
}

#endif
