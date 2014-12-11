#ifndef GaLG_container_raw_data_h
#define GaLG_container_raw_data_h

#include <vector>
#include <string>

namespace GaLG {
  namespace container {
    using namespace std;
    struct raw_data {
      int num_of_instances;
      int num_of_attributes;

      vector<string> meta;
      vector<vector<string> > instance;

      raw_data& select(string, vector<string>&);
      raw_data& select(int, vector<string>&);
    };
  }
}

#endif
