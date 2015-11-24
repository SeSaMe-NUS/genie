#ifndef GPUGenie_h
#define GPUGenie_h

extern bool GPUGENIE_ERROR;
extern unsigned long long GPUGENIE_TIME;

#ifndef GPUGENIE_DEBUG
#define GPUGENIE_DEBUG
#endif

//#if defined(GPUGENIE_DEBUG) && !defined(DEBUG_VERBOSE)
//#define DEBUG_VERBOSE
//#endif

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct data_{
  u32 id;
  float aggregation;
} data_t;

//for ide: to revert it as system file later, change <> to ""
#include <GPUGenie/raw_data.h>
#include <GPUGenie/inv_list.h>
#include <GPUGenie/inv_table.h>
#include <GPUGenie/query.h>
#include <GPUGenie/match.h>
#include <GPUGenie/topk.h>
#include <GPUGenie/parser/parser.h>
#include <GPUGenie/interface.h>
#include <GPUGenie/knn.h>
#include <GPUGenie/FileReader.h>// to read csv file data for simple examples


#endif
