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

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/detail/config.h>
#include <thrust/random.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/equal.h>
#include <thrust/sort.h>

#if (ENABLE_ISPC)

#include <libhu/timing.h>
#include <libhu/cpuid.h>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning (disable: 4244)
#pragma warning (disable: 4305)
#endif

#include <libhu/detail/backend/ispc/detail/hash_utils_ispc.h>

#endif

#include <libhu/detail/backend/hash_utils.h>

#if (ENABLE_NVPA)
#include <NvPAUtil.h>
NVPA_EXTERN_GLOBALS;
#endif

//#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
//  NVPA_DECLARE_KERNEL_EVENT_MEMORY;
//#else
//#ifndef NVPA_DECLARE_KERNEL_EVENT_MEMORY
//  AGPM_EXTERN_KERNEL_EVENT_MEMORY;
//#endif
//#endif

#define DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION 1

// ------------------------------------------------------------------

//#define OFFSETS_TABLE_16 {0u,1153073u,35879u,929807u,1299817u,674431u,208291u,1159789u,17681u,101449u,679681u,513269u,515701u,186379u,946961u,690493u}
//#define OFFSETS_TABLE_16 {0u,104729u,54781u,43121u,61109u,8201u,12461u,1505u,47589u,61433u,30813u,33873u,43541u,63721u,23821u,37825u}
#define OFFSETS_TABLE_16 {0u,3949349u,8984219u,9805709u,7732727u,1046459u,9883879u,4889399u,2914183u,3503623u,1734349u,8860463u,1326319u,1613597u,8604269u,9647369u}

// ------------------------------------------------------------------

CUDA_CONST libhu::U32 offsets[16] = OFFSETS_TABLE_16;

// ------------------------------------------------------------------

#if (__CUDACC__)

__inline__ __device__ libhu::U64 atomicMaxU64(libhu::U64* address, libhu::U64 val)
{
  libhu::U64 old = *address, assumed;
  if (*address < val)
  {
    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, max(val,assumed) );
    }
    while (assumed != old);
  }
  return old;
}

__inline__ __device__  libhu::U32 atomicMaxU32(libhu::U32* address, libhu::U32 val)
{
  return atomicMax(address, val);
} 

#else

__inline__ __host__ libhu::U64 atomicMaxU64(libhu::U64* address, libhu::U64 val)
{
  libhu::U64 old = address[0];
  address[0]     = std::max<libhu::U64>(address[0], val);
  return old;
}

//__inline__ __host__ libhu::U32 atomicMaxU32(libhu::U32* address, libhu::U32 val)
//{
//  libhu::U32 old = address[0];
//  address[0] = std::max<libhu::U32>(address[0], val);
//  return old;
//}

__inline__ __host__ libhu::U32 atomicMaxU32(libhu::U32* address, libhu::U32 val)
{
  //ispc::atomicMaxU32((libhu::U32)address, val);
  return 1;
}

#endif

namespace libhu
{

// ------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         U32 KEY_TYPE_MASK>
    __host__ __device__
    bool is_hashed_functor(T_KEY a, T_HASH_TABLE b)
    {
      return ((b & KEY_TYPE_MASK) == a);
    }

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         U32 KEY_TYPE_MASK>
         bool check_access(T_RAND_ACCESS_ITERATOR1 keys_begin, T_RAND_ACCESS_ITERATOR1 keys_end, T_RAND_ACCESS_ITERATOR2 output_values_begin)
         {
           typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
           typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_HASH_TABLE;
           return thrust::equal(keys_begin, keys_end, output_values_begin, is_hashed_functor<T_KEY, T_HASH_TABLE, KEY_TYPE_MASK>);
         }
         
//------------------------------------------------------------------------

template <typename T>
struct notEqualToUndefinedPackedKey : public thrust::unary_function<T,T>
{
  T undefinedKey;

  __host__ __device__
  notEqualToUndefinedPackedKey(T _undefinedKey) : 
    undefinedKey(_undefinedKey)
    {
    }

  __host__ __device__
  bool operator()(T packedKey)
  {
    return (packedKey != undefinedKey);
  }
};

// ------------------------------------------------------------------
        
template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_HASH_FUNCTOR>
bool check_hashed(T_RAND_ACCESS_ITERATOR1 hash_table_begin, T_RAND_ACCESS_ITERATOR1 hash_table_end, U32 num_keys, T_HASH_FUNCTOR hf)
{
 typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_HASH_TABLE;
 libhu::U32 num_hashed_keys = thrust::count_if(hash_table_begin, hash_table_end, notEqualToUndefinedPackedKey<T_HASH_TABLE>(hf.PACKED_UNDEFINED_KEY));

#if (ENABLE_LIBHU_LOG)
 std::cerr << "num_keys                         : " << num_keys << std::endl;
 std::cerr << "num_hashed_keys                  : " << num_hashed_keys << std::endl;
#endif

 return (num_hashed_keys == num_keys);
 
}

//------------------------------------------------------------------------

template <typename iterator>
void print(const std::string& name, iterator first, iterator last)
{
  typedef typename std::iterator_traits<iterator>::value_type t;

  std::cerr << name << ": ";
  thrust::copy(first, last, std::ostream_iterator<t>(std::cerr, " "));
  std::cerr << std::endl;
}

//------------------------------------------------------------------------

template <typename T_HASH_TABLE,
          typename T_HASH_FUNCTOR>
struct print_key
{
  __host__ __device__
  print_key(T_HASH_FUNCTOR _hf): hf(_hf)
  {
  }

  __host__ __device__
  void operator()(T_HASH_TABLE t)
  {
    std::cerr << hf.GET_KEY_POS(t) << " ";
  }
  
  T_HASH_FUNCTOR hf;
};

//------------------------------------------------------------------------

template <typename T_RAND_ACCES_ITERATOR1,
          typename T_HASH_FUNCTOR>
void print_keys(T_RAND_ACCES_ITERATOR1 hash_table_begin, T_RAND_ACCES_ITERATOR1 hash_table_end, T_HASH_FUNCTOR hf)
{
  typedef typename std::iterator_traits<T_RAND_ACCES_ITERATOR1>::value_type T_HASH_TABLE;
  thrust::for_each(hash_table_begin, hash_table_end, print_key<T_HASH_TABLE, T_HASH_FUNCTOR>(hf));
  std::cerr << std::endl;
}

//------------------------------------------------------------------------

template <typename T_HASH_TABLE,
          typename T_HASH_FUNCTOR>
struct print_value
{
  __host__ __device__
  print_value(T_HASH_FUNCTOR _hf): hf(_hf)
  {
  }

  __host__ __device__
  void operator()(T_HASH_TABLE t)
  {
    std::cerr << hf.GET_KEY_ATTACH_ID(t) << " ";
  }
  
  T_HASH_FUNCTOR hf;
};

//------------------------------------------------------------------------

//bool checkTargetISAIsSupported() {
//  bool isaSupported = false;
//#if defined(ISPC_TARGET_SSE2)
//  isaSupported = CPUSupportsSSE2();
//#elif defined(ISPC_TARGET_SSE4)
//  isaSupported = CPUSupportsSSE4();
//#elif defined(ISPC_TARGET_AVX)
//  isaSupported = CPUSupportsAVX();
//#endif
//  if (!isaSupported) {
//    return true;
//  }
//  else
//  {
//    return false;
//  }
//}

//------------------------------------------------------------------------

template <typename T_RAND_ACCES_ITERATOR1,
          typename T_HASH_FUNCTOR>
void print_values(T_RAND_ACCES_ITERATOR1 hash_table_begin, T_RAND_ACCES_ITERATOR1 hash_table_end, T_HASH_FUNCTOR hf)
{
  typedef typename std::iterator_traits<T_RAND_ACCES_ITERATOR1>::value_type T_HASH_TABLE;
  thrust::for_each(hash_table_begin, hash_table_end, print_value<T_HASH_TABLE, T_HASH_FUNCTOR>(hf));
  std::cerr << std::endl;
}

// ------------------------------------------------------------------

//This function initializes the 512bit data according to the MD5 spec. 
//Such that, the first 128 bit is the input;
//we also xor these 128 bits with the key, which can act like a seed value. 
//And the rest up of the 12 32bit data blocks are filled
//according to the md5 spec, in order to pad our data to 512 bits.
//block 0-3: input xor with key
//block 4: 0x80000000. This correponds to append 1 bit to block 0-4.
//block 5-13: 0. This corresponds to appending zeros up to 448 bit.
//block 14-15: 0x0000000000000080. This correspond to the bit length of the input (128 bit), as a 64bit 
//litten endian.
__inline__ __host__ __device__ 
void setupInput(UVEC4 _input, U32 key, U32 data[16])
{
  data[0]  = _input.x^key; data[1] = _input.y^key; data[2] = _input.z^key; data[3] = _input.w^key; //xor base with key
  data[4]  = 0x80000000u;
  data[5]  = 0u; data[6]  = 0u; data[7]  = 0u; data[8]  = 0u; 
  data[9]  = 0u; data[10] = 0u; data[11] = 0u; data[12] = 0u; data[13] = 0u;
  data[14] = 0x00000000u; data[15]=0x00000080u; 
}

// ------------------------------------------------------------------

//initialize to the 4 hexes.
__inline__ __host__ __device__ 
UVEC4 initDigest()
{
  return make_uint4(0x01234567u,0x89ABCDEFu,0xFEDCBA98u,0x76543210u);
}

// ------------------------------------------------------------------

//F compression functions
//(b & c) | ((not b) & d)
__inline__ __host__ __device__ 
U32 F0_15(UVEC3 tD)
{
  return (tD.x & tD.y) | ((~tD.x) & tD.z);
}

// ------------------------------------------------------------------

//(d & b) | ((not d) & c)
__inline__ __host__ __device__ 
U32 F16_31(UVEC3 tD)
{
  return (tD.z & tD.x) | ((~tD.z) & tD.y);
}

// ------------------------------------------------------------------

//b ^ c ^ d
__inline__ __host__ __device__ 
U32 F32_47(UVEC3 tD)
{
  return tD.x ^ tD.y ^ tD.z;
}

// ------------------------------------------------------------------

//c ^ (b | (~d))
__inline__ __host__ __device__ 
U32 F48_63(UVEC3 tD)
{
  return tD.y ^ (tD.x | (~tD.z));
}

// ------------------------------------------------------------------

//const U32 WN_RAND_MAX = 4294967295;
CUDA_CONST
U32 WN_RAND_MAX = U32(429467294);

__inline__ __host__ __device__ 
float conv(U32 _input)
{
  return float(_input) / (float(WN_RAND_MAX)+1.0);
}

// ------------------------------------------------------------------

__inline__ __host__ __device__ 
FVEC4 convertToR0_R1(UVEC4 _input)
{
  FVEC4 _output;
  _output.x = conv(_input.x);
  _output.y = conv(_input.y);
  _output.z = conv(_input.z);
  _output.w = conv(_input.w);
  
  return _output;
}

// ------------------------------------------------------------------

__inline__ __host__ __device__ 
UVEC4 whiteNoise(UVEC4 _input, U32 key)
{
  U32 data[16];
  setupInput(_input,key,data);
  UVEC4 rot0_15 = make_uint4(7u,12u,17u,22u);
  UVEC4 rot16_31 = make_uint4(5u,9u,14u,20u);
  UVEC4 rot32_47 = make_uint4(4u,11u,16u,23u);
  UVEC4 rot48_63 = make_uint4(6u,10u,15u,21u);
  
  UVEC4 digest = initDigest();
  UVEC4 tD;
  //UVEC4 fTmp;
  U32 fTmp;
  U32 i = 0u;
  U32 idx;
  U32 r;
  U32 trig; 
  //const U32 MAXFT = 4294967295; 
  //2^32-1
  //U32 MAXFT = 4294967000;
  U32 MAXFT = U32(429467294);
  //What follows is the unrolled loop from 0 through 63
  //0
  tD = digest;
  
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //1
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //2
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //3
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
        trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //4
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //5
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //6
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //7
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //8
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //9
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //10
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //11
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //12
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //13
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //14
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //15
  fTmp = F0_15(make_uint3(tD.y,tD.z,tD.w));
  idx = i++;
  r = rot0_15.x;
  rot0_15 = make_uint4(rot0_15.y,rot0_15.z,rot0_15.w,rot0_15.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  
  //16-31  
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //17
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //18
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //19
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //20
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //21
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //22
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //23
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //24
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //25
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //26
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //27
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //28
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //29
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //30
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //31
  fTmp = F16_31(make_uint3(tD.y,tD.z,tD.w));
  idx = (5u*i++ + 1u) % 16u;
  r = rot16_31.x;
  rot16_31 = make_uint4(rot16_31.y,rot16_31.z,rot16_31.w,rot16_31.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;

  //32-47
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //33
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //34
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //35
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //36
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //37
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //38
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //39
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //40
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //41
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //42
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //43
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //44
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //45
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //46
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //47
  fTmp = F32_47(make_uint3(tD.y,tD.z,tD.w));
  idx = (3u*i++ + 5u) % 16u;
  r = rot32_47.x;
  rot32_47 = make_uint4(rot32_47.y,rot32_47.z,rot32_47.w,rot32_47.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  
  //48-63
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //49
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //50
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //51
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //52
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //53
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //54
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //55
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //56
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //57
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //58
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //59
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //60
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //61
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //62
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  //63
  fTmp = F48_63(make_uint3(tD.y,tD.z,tD.w));
  idx = (7u*i++) % 16u;
  r = rot48_63.x;
  rot48_63 = make_uint4(rot48_63.y,rot48_63.z,rot48_63.w,rot48_63.x);
  trig = U32(floor(abs(sin(float(i)))*float(MAXFT)));
  tD.x = tD.y + ((tD.x+fTmp+data[int(idx)]+trig) << r);
  tD = make_uint4(tD.y,tD.z,tD.w,tD.x);
  digest.x += tD.x;
  digest.y += tD.y;
  digest.z += tD.z;
  digest.w += tD.w;
  
  return digest;   
} 

//------------------------------------------------------------------------

template <typename TYPE>
struct constant_forward_offset : public thrust::unary_function<TYPE,TYPE>
{
  __host__ __device__
  constant_forward_offset(U32 _offset_range) : offset_range(_offset_range)
  {} 

  __host__ __device__
  TYPE operator()(TYPE elem)
  {
    return (elem * offset_range);
  }
  U32 offset_range;
};

//------------------------------------------------------------------------

template<typename T>
struct overflow_count
{
  __host__ __device__
  overflow_count(U32 _max): max(_max)
  {
  }
  
  __host__ __device__
  bool operator()(T t) const 
  { 
    return (t > max); 
  }
  
  U32 max;
};

//------------------------------------------------------------------------

template <typename TYPE>
struct generate_unique_random_functor : public thrust::unary_function<TYPE,TYPE>
{
  __host__ __device__
  generate_unique_random_functor(U32 _offset_range, U32 _seed) : 
    offset_range(_offset_range),
    seed(_seed)
  {} 

  __host__ __device__
  TYPE operator()(TYPE elem)
  {
    UVEC4 relem = whiteNoise(make_uint4(elem, elem, elem, elem), seed);

    TYPE o = relem.x % offset_range;
    return ((elem * offset_range) + o);
  }
  
  U32 offset_range;
  U32 seed;
};

// ------------------------------------------------------------------

template <typename T_RAND_ACCESS_ITERATOR>
bool generate_unique_random(T_RAND_ACCESS_ITERATOR bufferBegin, T_RAND_ACCESS_ITERATOR bufferEnd, libhu::U32 min_range, libhu::U32 max_range, libhu::U32 seed) 
{
   libhu::U32 size         = bufferEnd - bufferBegin;
   libhu::S32 offset_range = max_range / (size + 1);
   // Avoid power of two offset_range, to avoid aligned patterns in the generated random values
   offset_range     = IS_POW_2<libhu::S32>(offset_range) ? offset_range - 1 : offset_range;
   offset_range     = max(1, offset_range);

   std::cerr << "offset_range:   " << offset_range << std::endl;
   
   typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR>::type TYPE;
   libhu::UPTR bufferUPtr = (libhu::UPTR)thrust::raw_pointer_cast(&*bufferBegin);
  
   thrust::counting_iterator<libhu::U32> countBegin(min_range);
   thrust::counting_iterator<libhu::U32> countEnd = countBegin + size;
   thrust::transform(countBegin, countEnd, bufferBegin, generate_unique_random_functor<TYPE>(offset_range, seed));    
  
   thrust::host_vector<TYPE> host_buffer(size);
   thrust::copy(bufferBegin, bufferEnd, host_buffer.begin());  
   bool unique = (thrust::unique(host_buffer.begin(), host_buffer.end()) == host_buffer.end());
   
    //unsigned int kMin = 0xFFFFFFFF, kMax = 0;
    //// Copy the first numElements keys into h_keysInsert
    //for (size_t i = 0; i < size; ++i)
    //{
    //    if (host_buffer[i] < kMin) kMin = host_buffer[i];
    //    if (host_buffer[i] > kMax) kMax = host_buffer[i];
    //}
    //std::cerr << "kMin:" << kMin << std::endl;
    //std::cerr << "kMax:" << kMax << std::endl;
    //double kAvgDist = 0;
    //unsigned int k = 0;
    //for (size_t i = 0; i < size; ++i)
    //{
    //  for (size_t j = 0; j < size; ++j)
    //  {
    //    if (i != j)
    //    {
    //      k++;
    //      kAvgDist += (max(host_buffer[i],host_buffer[j]) - min(host_buffer[i],host_buffer[j]));
    //    }
    //  }
    //}
    //kAvgDist /= k;
    //std::cerr << "kAvgDist:" << kAvgDist << std::endl;

   // error: non-unique keys detected
   std::cerr << "unique:         " << unique << std::endl;
   assert(unique);
   
   U32 num_keys_overflows = thrust::count_if(host_buffer.begin(), host_buffer.end(), overflow_count<TYPE>(max_range));
   bool in_range = (num_keys_overflows == 0);
   std::cerr << "keys_overflows: " << num_keys_overflows << std::endl;
   // error: keys out of range detected
   assert(in_range);
   
   srand(seed);
   thrust::host_vector<TYPE> host_keys(size); 
   thrust::copy(bufferBegin, bufferEnd, host_keys.begin());
   std::random_shuffle(host_keys.begin(), host_keys.end());
   thrust::copy(host_keys.begin(), host_keys.end(), bufferBegin);   
   
   return (unique && in_range);
  
}

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
              T_HASH_FUNCTOR          hf)
{
  bool constrained_hash_access = false;
  libhu::U32 max_age = hf.KEY_TYPE_MAX_AGE;
  libhu::detail::backend::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
}

//------------------------------------------------------------------------

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
              libhu::U32              max_age)
{
  bool constrained_hash_access = false;
  libhu::detail::backend::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
}

//------------------------------------------------------------------------

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
              bool                    constrained_hash_access)
{
  libhu::U32 max_age = hf.KEY_TYPE_MAX_AGE;
  libhu::detail::backend::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
}

//------------------------------------------------------------------------

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
              libhu::U32              max_age)
{
  libhu::detail::backend::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
}

//------------------------------------------------------------------------

} // end namespace libhu

