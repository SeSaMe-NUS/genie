#include "match_common.h"

#include "match_inlines.cu"

namespace genie
{
namespace core
{

__global__
void convert_to_data(T_HASHTABLE* table, u32 size)
{
    u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
        return;
    data_t *mytable = reinterpret_cast<data_t*>(&table[tid]);
    u32 agg = get_key_attach_id(table[tid]);
    u32 myid = get_key_pos(table[tid]);
    mytable->id = myid;
    mytable->aggregation = *reinterpret_cast<float*>(&agg);
}

} // namespace core

} // namespace genie

