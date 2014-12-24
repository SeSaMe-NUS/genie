#ifndef GaLG_raw_data_h
#define GaLG_raw_data_h

#include <vector>
#include <string>

using namespace std;

namespace GaLG {
  class raw_data {
    private:
      vector<string> meta;
      vector<vector<string> > instance;
      vector<vector<string> > transpose;

    public:
      void clear();
      int set_meta(vector<string>&);
      int set_meta(int, string);
      int set_meta(string, string);
      int get_meta(int, string&);
      string* get_meta(int);
      int m_size();
      int i_size();
      int add_row(vector<string>&);
      int get_row(int, vector<string>&);
      vector<string>* get_row(int);
      int select(string, vector<string>&);
      int select(int, vector<string>&);
      vector<string>* select(string);
      vector<string>* select(int);
  };
}

#endif
