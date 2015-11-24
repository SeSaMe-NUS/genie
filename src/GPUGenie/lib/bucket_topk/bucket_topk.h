#ifndef BUCKET_TOPK_H_
#define BUCKET_TOPK_H_

#define THREADS_PER_BLOCK 256
#define BUCKET_TOPK_MAX_K 1000
#define rpc(X) raw_pointer_cast((X).data())

#define BUCKET_TOPK_GAP 1
#define BUCKET_TOPK_EQUAL(X, Y) ((X-Y)<BUCKET_TOPK_GAP && (X-Y)>0-BUCKET_TOPK_GAP)

typedef struct Bucket {
	int a_index;
	int b_index;
} Bucket;

#include "utils.inc"
#include "update_min_max.inc"
#include "count_sum.inc"
#include "assign_bucket.inc"
#include "result_thread_partition.inc"
#include "save_result.inc"
#include "bucket_topk.inc"
#endif
