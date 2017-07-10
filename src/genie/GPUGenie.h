#ifndef GPUGenie_h
#define GPUGenie_h

typedef unsigned int u32;
typedef unsigned long long u64;

#include <genie/table/inv_list.h>
#include <genie/table/inv_table.h>
#ifdef COMPR
    #include <genie/table/inv_compr_table.h>
#endif
#include <genie/query/query.h>
#include <genie/matching/match.h>
#include <genie/matching/heap_count.h>
#include <genie/utility/FileReader.h>
#include <genie/interface/interface.h>
#include <genie/matching/knn.h>


#include <genie/utility/Logger.h>
#include <genie/utility/Timing.h>
#include <genie/exception/genie_errors.h>

#endif
