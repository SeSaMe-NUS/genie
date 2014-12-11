#include "parser.h"

#include <vector>
#include <string>
#include <stdio.h>
#include "GaLG/lib/libcsv/csv.h"

using namespace std;

struct csv_info {
  size_t rows;
  vector<string>& meta;
  vector<string>& data;
};

static void cb1(void *s, size_t len, void *data)
{
  struct csv_info *info = (struct csv_info *)data;
  if(info -> rows == 0)
    info -> meta.push_back(string((const char*)s));
  else
    info -> data.push_back(string((const char*)s));
}

static void cb2(int c, void *data)
{
  (((struct csv_info *)data)->rows)++;
}

static int is_space(unsigned char c)
{
  if (c == CSV_SPACE || c == CSV_TAB) return 1;
  return 0;
}

static int is_term(unsigned char c)
{
  if (c == CSV_CR || c == CSV_LF) return 1;
  return 0;
}

int GaLG::tool::csv(string file, raw_data& data)
{
  const char* f = file.c_str();
  vector<string> raw_meta;
  vector<string> raw_data;
  FILE* fp;
  struct csv_parser p;
  char buf[1024];
  size_t bytes_read;
  struct csv_info info = {0, raw_meta, raw_data};

  fp = fopen(f, "rb");
  if (!fp)
    return -1;

  if (csv_init(&p, CSV_APPEND_NULL) != 0)
    return -1;

  csv_set_space_func(&p, is_space);
  csv_set_term_func(&p, is_term);

  while ((bytes_read=fread(buf, 1, 1024, fp)) > 0)
  {
    if (csv_parse(&p, buf, bytes_read, cb1, cb2, &info) != bytes_read)
      return -1;
  }

  data.meta.clear(), data.meta.resize(raw_meta.size());
  data.instance.clear();
  copy(raw_meta.begin(), raw_meta.end(), data.meta.begin());
  for(size_t i=0; i<raw_data.size()/data.meta.size(); i++)
  {
    vector<string> tmp(data.meta.size());
    copy(raw_data.begin() + i*data.meta.size(), raw_data.begin() + (i+1)*data.meta.size(), tmp.begin());
    data.instance.push_back(tmp);
  }
  csv_fini(&p, cb1, cb2, &info);
  csv_free(&p);
  fclose(fp);

  return info.rows;
}
