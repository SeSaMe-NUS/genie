/*
 *  (C) copyright  2011, Ismael Garcia, (U.Girona/ViRVIG, Spain & INRIA/ALICE, France)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef KEY_COH_HASH_H_
#define KEY_COH_HASH_H_

#include <libhu/hash_utils.h>

// ------------------------------------------------------------------

class key_coh_hash_functor : libhu::key_hash_functor
{
public:

  key_coh_hash_functor()
  {
  }
  
typedef libhu::U32 T_KEY;
typedef libhu::U32 T_AGE;
typedef libhu::U32 T_MAX_AGE;
typedef libhu::U32 T_LOC;
typedef libhu::U32 T_HASH_TABLE;

static const libhu::U32 DEFAULT_GROUP_SIZE       = 192u;

static const libhu::U32 KEY_TYPE_BITS            = 28u;
static const libhu::U32 KEY_TYPE_MASK            = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1u );
static const libhu::U32 PACKED_KEY_TYPE_MASK     = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1u );
static const libhu::U32 KEY_TYPE_RANGE           = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 2u );
static const libhu::U32 UNDEFINED_KEY            = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1u );
static const libhu::U32 PACKED_UNDEFINED_KEY     = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1u );

static const libhu::U32 KEY_TYPE_AGE_MASK        = 15u;
static const libhu::U32 KEY_TYPE_AGE_BITS        = 4u;
static const libhu::U32 KEY_TYPE_INIT_AGE        = 1u;
static const libhu::U32 KEY_TYPE_NULL_AGE        = 0u;
static const libhu::U32 KEY_TYPE_MAX_AGE         = 16u;
static const libhu::U32 KEY_TYPE_MAX_AGE_MASK    = 4u;
static const libhu::U32 KEY_TYPE_MAX_AGE_BITS    = 4u;

static const libhu::U32 HTABLE_ID                = 0u;
static const libhu::U32 NOP_MODE_TRUE            = 255u;
static const libhu::U32 NOP_MODE_FALSE           = 0u;

libhu::UPTR hash_tableUPtr;
libhu::U32* max_tableUPtr;
libhu::U32  hash_table_size;

__inline__ __host__ __device__ T_LOC        WRAP(T_LOC A, T_LOC B)                          { return ((A) % (B)); }

__inline__ __host__ __device__ T_KEY        GET_KEY_POS(T_HASH_TABLE k)                     { return ((k) & KEY_TYPE_MASK); }
__inline__ __host__ __device__ T_KEY        GET_KEY_ATTACH_ID(T_HASH_TABLE k)               { return ((k) & KEY_TYPE_MASK); }
__inline__ __host__ __device__ T_AGE        GET_KEY_AGE(T_HASH_TABLE k)                     { return ((k) >> KEY_TYPE_BITS); }
__inline__ __host__ __device__ T_MAX_AGE    GET_KEY_MAX_AGE(T_HASH_TABLE k)                 { return ((k) >> KEY_TYPE_BITS); }
__inline__ __host__ __device__ T_HASH_TABLE PACK_KEY_POS(T_KEY p)                           { return ((p) & KEY_TYPE_MASK); }
__inline__ __host__ __device__ T_HASH_TABLE PACK_KEY_POS_AND_AGE(T_KEY p, T_AGE a)          { return (((a << KEY_TYPE_BITS)) + (p & KEY_TYPE_MASK)); }
__inline__ __host__ __device__ T_HASH_TABLE PACK_KEY_POS_AND_MAX_AGE(T_KEY p, T_MAX_AGE m)  { return (((m << KEY_TYPE_BITS)) + (p & KEY_TYPE_MASK)); }


// Hash function
__inline__ __host__ __device__ T_LOC h(T_KEY K, T_AGE AGE, libhu::U32 HSZ)
{
  return WRAP((offsets[AGE] + K), HSZ);
}

// Max. age operator to update hash_table
__device__ 
void operator()(T_HASH_TABLE& t)
{
  libhu::U32 i = (((libhu::UPTR)thrust::raw_pointer_cast(&t)) - ((libhu::UPTR)hash_tableUPtr)) / (sizeof(T_HASH_TABLE));
  if (t != PACKED_UNDEFINED_KEY)
  {
    t = PACK_KEY_POS_AND_MAX_AGE(GET_KEY_POS(t), max_tableUPtr[i]);
  }
}

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR,
         typename T_MAX_AGE_COMPUTATION_FUNCTOR>
__inline__ __host__ __device__  
void hash_kernel(libhu::U32            keys_size,
                 libhu::U32            hash_table_size,
                 T_KEY          keys[],
                 T_HASH_TABLE   hash_table[],
                 T_MAX_AGE      max_table[],
                 T_HASH_FUNCTOR hf,
                 T_MAX_AGE_COMPUTATION_FUNCTOR maf)
{

  // initialize variables
  libhu::U32 LOC;
  libhu::U32 ROOT_LOC;
  libhu::U8 AGE            = KEY_TYPE_NULL_AGE;  
  T_HASH_TABLE EVICTED_PKEY;
  T_HASH_TABLE PKEY = (GTID < keys_size) ? PACK_KEY_POS_AND_MAX_AGE(keys[ (GTID) ], KEY_TYPE_INIT_AGE) : PACKED_UNDEFINED_KEY;
  AGE               = (GTID < keys_size) ? KEY_TYPE_NULL_AGE : KEY_TYPE_MAX_AGE;

  while (AGE < KEY_TYPE_MAX_AGE)
  {
      LOC  = h(GET_KEY_POS(PKEY), AGE, hash_table_size);
      EVICTED_PKEY = atomicMaxU32(&hash_table[ (LOC) ], PKEY );

      if (EVICTED_PKEY < PKEY)
      { 

        maf.update_max_age(hash_table_size, PKEY, AGE, max_table, hf);

        if (GET_KEY_AGE(EVICTED_PKEY) > 0u)
        { 
          PKEY       = EVICTED_PKEY;
          AGE        = GET_KEY_AGE(EVICTED_PKEY); 
        }
        else 
        {
          break;
       }
    }
    else
    {
      AGE++;
      PKEY     = PACK_KEY_POS_AND_MAX_AGE(GET_KEY_POS(PKEY), AGE);
    }
  }

}
    
};

#endif