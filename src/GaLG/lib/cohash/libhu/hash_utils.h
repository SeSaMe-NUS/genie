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

#pragma once

#include <iostream>
#include <string>

// ------------------------------------------------------------------
namespace libhu
{

// ------------------------------------------------------------------

#include <vector_functions.h>

typedef unsigned char          U8;
typedef unsigned short         U16;
typedef unsigned int           U32;
typedef signed char            S8;
typedef signed short           S16;
typedef signed int             S32;
typedef float                  F32;
typedef double                 F64;

typedef uint2                  UVEC2;
typedef uint3                  UVEC3;
typedef uint4                  UVEC4;
typedef int2                   IVEC2;
typedef int3                   IVEC3;
typedef int4                   IVEC4;
typedef float2                 FVEC2;
typedef float3                 FVEC3;
typedef float4                 FVEC4;

#if (__CUDACC__)

typedef unsigned long long int U64;
typedef signed long long       S64;

#define CUDA_CONST             __constant__
#define CUDA_CONST_VAR         const

#else

typedef unsigned __int64       U64;
typedef signed __int64         S64;

#define CUDA_CONST             static const
#define CUDA_CONST_VAR         const

#endif

#if (_M_X64) || (__x86_64__)
typedef S64                    SPTR;
typedef U64                    UPTR;
#else
typedef __w64 S32              SPTR;
typedef __w64 U32              UPTR;
#endif

// ------------------------------------------------------------------

template<typename T_VALUE>
__inline__ __host__ __device__ 
T_VALUE IS_POW_2(T_VALUE value) { return (!(value & (value - 1)) && value); }

// ------------------------------------------------------------------

#if (__CUDACC__)

#define BARRIER             __syncthreads()
#define TID                 threadIdx.x
#define TBID                blockIdx.x
#define TBSZ                blockDim.x
//#define GTID                (threadIdx.x + blockIdx.x * blockDim.x)
#define GTID                (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x)
//#define NUM_INPUT_KEYS      (blockDim.x * gridDim.x)
#define NUM_INPUT_KEYS      (blockDim.x * blockDim.y * gridDim.x)
#define NUM_SET_OF_KEYS     TBSZ
#define SET_OF_KEYS_ID      TBID

#endif

// ------------------------------------------------------------------

struct constrained_hash_access_tag 
{

  template<typename T_HASH_TABLE,
           typename T_MAX_AGE,
           typename T_HASH_FUNCTOR>
  __inline__ __host__ __device__
  void update_max_age(libhu::U32     hash_table_size,
                      T_HASH_TABLE   PKEY,
                      libhu::U8      AGE,
                      T_MAX_AGE      max_table[],
                      T_HASH_FUNCTOR hf
                      )
  {
  }

  template<typename T_HASH_TABLE,
           typename T_HASH_FUNCTOR>
 __inline__ __host__ __device__
  void get_max_age(T_HASH_TABLE   OUT_KEY,
                   libhu::U8      max_age,
                   libhu::U8     &MAX_AGE,
                   T_HASH_FUNCTOR hf)
  {
    MAX_AGE     = max_age;
  }

};

struct unconstrained_hash_access_tag 
{

  template<typename T_HASH_TABLE,
           typename T_MAX_AGE,
           typename T_HASH_FUNCTOR>
  __inline__ __host__ __device__
  void update_max_age(libhu::U32     hash_table_size,
                      T_HASH_TABLE   PKEY,
                      libhu::U8      AGE,
                      T_MAX_AGE      max_table[],
                      T_HASH_FUNCTOR hf
                      )
  {
    libhu::U32 ROOT_LOC = hf.h(hf.GET_KEY_POS(PKEY), 0u, hash_table_size);
    atomicMaxU32(&max_table[ (ROOT_LOC) ], hf.GET_KEY_AGE(PKEY));
  }

  template<typename T_HASH_TABLE,
           typename T_HASH_FUNCTOR>
 __inline__ __host__ __device__
  void get_max_age(T_HASH_TABLE   OUT_KEY,
                   libhu::U8      max_age,
                   libhu::U8     &MAX_AGE,
                   T_HASH_FUNCTOR hf)
  {
    MAX_AGE = hf.GET_KEY_MAX_AGE(OUT_KEY);
  }

};

// ------------------------------------------------------------------

class key_hash_functor
{
public:

};

// ------------------------------------------------------------------

template <typename T_KEY,
          U32 KEY_BITS> 
class key_1d
{

public:

U32 w;

static const U32 MAX_1D_UNIVERSE_RANGE_W  = ( (1u << (KEY_BITS)) );
static const U32 MAX_1D_UNIVERSE_RANGE_H  = ( (1u << (KEY_BITS)) );
static const U32 MAX_1D_UNIVERSE_RANGE_D  = 1u;
static const U32 U32_COORD_BITS           = (KEY_BITS);
static const U32 U32_COORD_RANGE          = ( (1u << U32_COORD_BITS) - 1u );
static const U32 U32_COORD_OUT_RANGE      = ( (1u << U32_COORD_BITS) );

__inline__ __host__ __device__ U32   GET_1D_ALIGNMENT(T_KEY v) { return (v % w); }

key_1d(U32 _w) : w(_w)
{
}

key_1d() : w(MAX_1D_UNIVERSE_RANGE_W)
{
}

};

// ------------------------------------------------------------------

template <typename T_KEY,
          U32 KEY_BITS> 
class key_2d
{

public:

U32 w;
U32 h;
static const U32 MAX_2D_UNIVERSE_RANGE_W  = ( (1u << (KEY_BITS / 2u)) );
static const U32 MAX_2D_UNIVERSE_RANGE_H  = ( (1u << (KEY_BITS / 2u)) );
static const U32 MAX_2D_UNIVERSE_RANGE_D  = 1u;
static const U32 UVEC2_COORD_BITS         = (KEY_BITS / 2u);
static const U32 UVEC2_COORD_RANGE        = ( (1u << UVEC2_COORD_BITS) - 1u );
static const U32 UVEC2_COORD_OUT_RANGE    = ( (1u << UVEC2_COORD_BITS) );

__inline__ __host__ __device__ U32   GET_1D_ALIGNMENT(UVEC2 uk)               { return (((uk.y % h) * w) + (uk.x % w)); }
__inline__ __host__ __device__ UVEC2 GET_UVEC2_ALIGNMENT(T_KEY v)             { return make_uint2(v % w, (v / w) % h); }
__inline__ __host__ __device__ U32   GET_UVEC2_POS_X(T_KEY v)                 { return (((v) >> UVEC2_COORD_BITS) & UVEC2_COORD_RANGE); }
__inline__ __host__ __device__ U32   GET_UVEC2_POS_Y(T_KEY v)                 { return ((v) & UVEC2_COORD_RANGE); }
__inline__ __host__ __device__ T_KEY PACK_UVEC2(U32 x, U32 y)                 { return (((y & UVEC2_COORD_RANGE) << UVEC2_COORD_BITS) + (x & UVEC2_COORD_RANGE)); }
__inline__ __host__ __device__ UVEC2 UNPACK_UVEC2(T_KEY v)                    { return make_uint2(GET_UVEC2_POS_X(v), GET_UVEC2_POS_Y(v)); }

key_2d(U32 _w, U32 _h) : w(_w), h(_h) 
{
}

key_2d() : w(MAX_2D_UNIVERSE_RANGE_W), h(MAX_2D_UNIVERSE_RANGE_H) 
{
}

};

// ------------------------------------------------------------------

template <typename T_KEY,
          U32 KEY_BITS> 
class key_3d
{

public:

U32 w;
U32 h;
U32 d;

static const U32 MAX_3D_UNIVERSE_RANGE_W  = ( (1u << (KEY_BITS / 3u)) );
static const U32 MAX_3D_UNIVERSE_RANGE_H  = ( (1u << (KEY_BITS / 3u)) );
static const U32 MAX_3D_UNIVERSE_RANGE_D  = ( (1u << (KEY_BITS / 3u)) );
static const U32 UVEC3_COORD_BITS         = (KEY_BITS / 3u);
static const U32 UVEC3_COORD_RANGE        = ( (1u << UVEC3_COORD_BITS) - 1u );
static const U32 UVEC3_COORD_OUT_RANGE    = ( (1u << UVEC3_COORD_BITS) );

__inline__ __host__ __device__ U32   GET_1D_ALIGNMENT(UVEC3 uk)                      { return (((uk.z % d) * (w*h)) + ((uk.y % h) * w) + ((uk.x % w))); }
__inline__ __host__ __device__ UVEC3 GET_UVEC3_ALIGNMENT(T_KEY v)                    { U32 z = (v / (w * h)); U32 y = ((v - (z * w * h)) / w); U32 x = (v - (z * w * h) - (y * w)); return make_uint3(x, y, z); }
__inline__ __host__ __device__ U32   GET_UVEC3_POS_X(T_KEY v)                        { return ( ((v) & UVEC3_COORD_RANGE) ); }
__inline__ __host__ __device__ U32   GET_UVEC3_POS_Z(T_KEY v)                        { return ( ((v) >> (UVEC3_COORD_BITS + UVEC3_COORD_BITS)) & UVEC3_COORD_RANGE ); }
__inline__ __host__ __device__ U32   GET_UVEC3_POS_Y(T_KEY v)                        { return ( ((v) >> UVEC3_COORD_BITS) & UVEC3_COORD_RANGE ); }
__inline__ __host__ __device__ T_KEY PACK_UVEC3(U32 x, U32 y, U32 z)                 { return ( ( ((z & UVEC3_COORD_RANGE) << (UVEC3_COORD_BITS + UVEC3_COORD_BITS)) + ( (y & UVEC3_COORD_RANGE) << UVEC3_COORD_BITS) + (x & UVEC3_COORD_RANGE) ) ); }
__inline__ __host__ __device__ UVEC3 UNPACK_UVEC3(T_KEY v)                           { return make_uint3(GET_UVEC3_POS_X(v), GET_UVEC3_POS_Y(v), GET_UVEC3_POS_Z(v)); }

key_3d(U32 _w, U32 _h, U32 _d) : w(_w), h(_h), d(_d) 
{
}

key_3d() : w(MAX_3D_UNIVERSE_RANGE_W), h(MAX_3D_UNIVERSE_RANGE_H), d(MAX_3D_UNIVERSE_RANGE_H)
{
}

};

// ------------------------------------------------------------------

template <typename T_ACCESS_ITERATOR>
void generate_unique_random(T_ACCESS_ITERATOR bufferBegin, T_ACCESS_ITERATOR bufferEnd, U32 offset_range, U32 min, U32 max, U32 seed);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_HASH_FUNCTOR>
         bool check_hashed(T_RAND_ACCESS_ITERATOR1 hash_table_begin, T_RAND_ACCESS_ITERATOR1 hash_table_end, U32 num_keys, T_HASH_FUNCTOR hf);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         U32 KEY_TYPE_MASK>
         bool check_access(T_RAND_ACCESS_ITERATOR1 keys_begin, T_RAND_ACCESS_ITERATOR1 keys_end, T_RAND_ACCESS_ITERATOR2 output_keys_begin);

// ------------------------------------------------------------------

template <typename iterator>
void print(const std::string& name, iterator first, iterator last);

// ------------------------------------------------------------------

template <typename T_RAND_ACCES_ITERATOR1,
          typename T_HASH_FUNCTOR>
void print_keys(T_RAND_ACCES_ITERATOR1 hash_table_begin, T_RAND_ACCES_ITERATOR1 hash_table_end, T_HASH_FUNCTOR hf);

// ------------------------------------------------------------------

template <typename T_RAND_ACCES_ITERATOR1,
          typename T_HASH_FUNCTOR>
void print_values(T_RAND_ACCES_ITERATOR1 hash_table_begin, T_RAND_ACCES_ITERATOR1 hash_table_end, T_HASH_FUNCTOR hf);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR>
  void access(T_RAND_ACCESS_ITERATOR1 search_keys_begin,
              T_RAND_ACCESS_ITERATOR1 search_keys_end,
              T_RAND_ACCESS_ITERATOR2 hash_table_begin,
              T_RAND_ACCESS_ITERATOR2 hash_table_end,
              T_RAND_ACCESS_ITERATOR3 output_values_begin,
              T_HASH_FUNCTOR          hf);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR>
  void access(T_RAND_ACCESS_ITERATOR1 search_keys_begin,
              T_RAND_ACCESS_ITERATOR1 search_keys_end,
              T_RAND_ACCESS_ITERATOR2 hash_table_begin,
              T_RAND_ACCESS_ITERATOR2 hash_table_end,
              T_RAND_ACCESS_ITERATOR3 output_values_begin,
              T_HASH_FUNCTOR          hf,
              libhu::U32              max_age);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR>
  void access(T_RAND_ACCESS_ITERATOR1 search_keys_begin,
              T_RAND_ACCESS_ITERATOR1 search_keys_end,
              T_RAND_ACCESS_ITERATOR2 hash_table_begin,
              T_RAND_ACCESS_ITERATOR2 hash_table_end,
              T_RAND_ACCESS_ITERATOR3 output_values_begin,
              T_HASH_FUNCTOR          hf,
              bool                    constrained_hash_access);

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR>
  void access(T_RAND_ACCESS_ITERATOR1 search_keys_begin,
              T_RAND_ACCESS_ITERATOR1 search_keys_end,
              T_RAND_ACCESS_ITERATOR2 hash_table_begin,
              T_RAND_ACCESS_ITERATOR2 hash_table_end,
              T_RAND_ACCESS_ITERATOR3 output_values_begin,
              T_HASH_FUNCTOR          hf,
              bool                    constrained_hash_access,
              libhu::U32              max_age);

// ------------------------------------------------------------------

} // end namespace libhu

// ------------------------------------------------------------------

#include <libhu/detail/hash_utils.inl>

