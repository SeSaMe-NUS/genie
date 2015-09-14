#ifndef GaLG_h
#define GaLG_h

extern bool GALG_ERROR;
extern unsigned long long GALG_TIME;

//#ifndef GALG_DEBUG
//#define GALG_DEBUG
//#endif

//#if defined(GALG_DEBUG) && !defined(DEBUG_VERBOSE)
//#define DEBUG_VERBOSE
//#endif

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct data_{
  u32 id;
  float aggregation;
} data_t;


#include <GaLG/raw_data.h>
#include <GaLG/inv_list.h>
#include <GaLG/inv_table.h>
#include <GaLG/query.h>
#include <GaLG/match.h>
#include <GaLG/topk.h>
#include <GaLG/parser/parser.h>
#include <GaLG/interface.h>
#include <GaLG/knn.h>


#endif
