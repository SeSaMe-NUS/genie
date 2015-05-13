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

#include <thrust/detail/config.h>

#if (ENABLE_NVPA)
#include <NvPAUtil.h>
NVPA_EXTERN_GLOBALS;
#endif

// ------------------------------------------------------------------

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/swap.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include <thrust/device_ptr.h>

#include <thrust/detail/mpl/math.h> // for log2<N>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/synchronize.h>
#include <thrust/detail/device/cuda/arch.h>

#if (ENABLE_NVPA)
//#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
  NVPA_DECLARE_KERNEL_EVENT_MEMORY;
//#else
//  AGPM_EXTERN_KERNEL_EVENT_MEMORY;
//#endif
#endif

#include <libhu/hash_utils.h>

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace libh
{
namespace detail
{

template<typename T>
  void destroy(T &x)
{
  x.~T();
} // end destroy()

namespace backend
{
namespace cuda
{
namespace detail
{

// ------------------------------------------------------------------

// define our own min() rather than #include <thrust/extrema.h>
template<typename T>
  __inline__ __host__ __device__
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end min()

// ------------------------------------------------------------------

static const libhu::U32 warp_size = 32;

// ------------------------------------------------------------------

template<libhu::U32 N>
  struct align_size_to_int
{
  static const libhu::U32 value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
__global__ 
void hash_kernel(libhu::U32                         keys_size,
                 libhu::U32                         hash_table_size,
                 T_KEY*                             keys,
                 T_HASH_TABLE*                      hash_table,
                 T_MAX_AGE*                         max_table,
                 T_HASH_FUNCTOR                     hf,
                 libhu::constrained_hash_access_tag constrained_hash_access)
{

  hf.hash_kernel(keys_size, hash_table_size, keys, hash_table, max_table, hf, constrained_hash_access);

}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
__global__ 
void hash_kernel(libhu::U32                           keys_size,
                 libhu::U32                           hash_table_size,
                 T_KEY*                               keys,
                 T_HASH_TABLE*                        hash_table,
                 T_MAX_AGE*                           max_table,
                 T_HASH_FUNCTOR                       hf,
                 libhu::unconstrained_hash_access_tag unconstrained_hash_access)
{

  hf.hash_kernel(keys_size, hash_table_size, keys, hash_table, max_table, hf, unconstrained_hash_access);

}

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_HASH_FUNCTOR>
  void hash(T_RAND_ACCESS_ITERATOR1 keys_begin,
            T_RAND_ACCESS_ITERATOR1 keys_end,
            T_RAND_ACCESS_ITERATOR2 hash_table_begin,
            T_RAND_ACCESS_ITERATOR2 hash_table_end,
            T_HASH_FUNCTOR          hf,
            bool                    constrained_hash_access,
            libhu::U32             &max_age)
{

  libhu::U32 block_size           = hf.DEFAULT_GROUP_SIZE;
  libhu::U32 keys_size            = keys_end - keys_begin;
  libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;
  libhu::U32 grid_size            = ceilf((libhu::F32)keys_size / (libhu::F32)block_size);
  libhu::U32 smem_size            = 0u;
       
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_HASH_TABLE;
  typedef typename thrust::device_vector<libhu::U32>::iterator T_RAND_ACCESS_ITERATOR4;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR4>::type T_MAX_AGE;
  
  libhu::UPTR max_table_dptr;
  libhu::UPTR hash_table_dptr;
  
  thrust::fill(hash_table_begin, hash_table_end, hf.PACKED_UNDEFINED_KEY);
  hash_table_dptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin)); 
  
  hf.hash_tableUPtr    = hash_table_dptr;
  hf.hash_table_size   = hash_table_size;
  
  libhu::U32 max_table_size = (!constrained_hash_access) ? hash_table_size : 1;
  thrust::device_vector<libhu::U32> max_table(max_table_size);
  thrust::fill(max_table.begin(), max_table.end(), hf.KEY_TYPE_NULL_AGE);  

  max_table_dptr   = (libhu::UPTR)(thrust::raw_pointer_cast(&*max_table.begin()));
  hf.max_tableUPtr = (libhu::U32*)max_table_dptr;  
  
  const unsigned int max1DBlocks = 65535;
  const unsigned int max2DBlocks = 65535u * 65535u;
  unsigned int       num_blocks   = grid_size;
  
  for (unsigned int block = 0; block < num_blocks; block += max2DBlocks)
  {
    unsigned int blocks          = min(max2DBlocks, num_blocks - block);
    unsigned int keys_size_range = min(blocks * block_size, keys_size - (block * block_size));
    
    // Round up in order to make sure all items are hashed in.
    const unsigned grid_size  = 16384;
    dim3 grid( (keys_size_range +block_size-1) / block_size );
    if (grid.x > grid_size) {
        grid.y = (grid.x + grid_size - 1) / grid_size;
        grid.x = grid_size;
    }

    libhu::UPTR keys_range_dptr   = (libhu::UPTR)(thrust::raw_pointer_cast(&*(keys_begin   + (block * block_size)) ));

#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
    NVPA_START_EVENT_GPU(cudakernelBuildHash);
    NVPA_GET_KERNEL_EVENT_MEMORY(grid.x * grid.y, block_size);
#endif

    if (constrained_hash_access)
    {
      libhu::constrained_hash_access_tag constrained_hash_access;
      hash_kernel
       <T_KEY, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
       <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_MAX_AGE*)max_table_dptr, hf, constrained_hash_access);
    }
    else
    {
      libhu::unconstrained_hash_access_tag unconstrained_hash_access;
      hash_kernel
       <T_KEY, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
       <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_MAX_AGE*)max_table_dptr, hf, unconstrained_hash_access);
    }
    thrust::detail::device::cuda::synchronize_if_enabled("hash_kernel");

#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
     NVPA_STOP_EVENT_GPU(cudakernelBuildHash);
#endif

  }

  if (constrained_hash_access)
  {
    // Compute global max. age
    //T_RAND_ACCESS_ITERATOR2 max_iter;
    //T_HASH_TABLE            key_and_age;
    //max_iter = thrust::max_element(hash_table_begin, hash_table_end);
    //thrust::copy(hash_table_begin + (max_iter - hash_table_begin), hash_table_begin + ((max_iter - hash_table_begin) + 1), &key_and_age);
    //max_age = hf.GET_KEY_AGE(key_and_age) + 1;
    // Set limit max. age
    max_age = hf.KEY_TYPE_MAX_AGE;
  }
  else
  {
    // Add max. chain to the hash table
    thrust::for_each(hash_table_begin, hash_table_end, hf);
  }

}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_VALUE,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
__global__ 
void hash_by_key_kernel(libhu::U32                         keys_size,
                        libhu::U32                         hash_table_size,
                        T_KEY*                             keys,
                        T_VALUE*                           values,
                        T_HASH_TABLE*                      hash_table,
                        T_MAX_AGE*                         max_table,
                        T_HASH_FUNCTOR                     hf,
                        libhu::constrained_hash_access_tag constrained_hash_access)
{
  hf.hash_by_key_kernel(keys_size, hash_table_size, keys, values, hash_table, max_table, hf, constrained_hash_access);
}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_VALUE,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
__global__ 
void hash_by_key_kernel(libhu::U32                           keys_size,
                        libhu::U32                           hash_table_size,
                        T_KEY*                               keys,
                        T_VALUE*                             values,
                        T_HASH_TABLE*                        hash_table,
                        T_MAX_AGE*                           max_table,
                        T_HASH_FUNCTOR                       hf,
                        libhu::unconstrained_hash_access_tag unconstrained_hash_access)
{
  hf.hash_by_key_kernel(keys_size, hash_table_size, keys, values, hash_table, max_table, hf, unconstrained_hash_access);
}

// ------------------------------------------------------------------

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR>
  void hash_by_key(T_RAND_ACCESS_ITERATOR1 keys_begin,
                   T_RAND_ACCESS_ITERATOR1 keys_end,
                   T_RAND_ACCESS_ITERATOR2 values_begin,
                   T_RAND_ACCESS_ITERATOR3 hash_table_begin,
                   T_RAND_ACCESS_ITERATOR3 hash_table_end,
                   T_HASH_FUNCTOR          hf,
                   bool                    constrained_hash_access,
                   libhu::U32             &max_age)
{
   
  libhu::U32 block_size           = hf.DEFAULT_GROUP_SIZE;
  libhu::U32 keys_size            = keys_end - keys_begin;
  libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;
  libhu::U32 num_blocks           = ceilf((libhu::F32)keys_size / (libhu::F32)block_size);
  libhu::U32 smem_size            = 0u;
       
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_VALUE;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR3>::type T_HASH_TABLE;
  typedef typename thrust::device_vector<libhu::U32>::iterator T_RAND_ACCESS_ITERATOR4;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR4>::type T_MAX_AGE;  
  
  libhu::UPTR max_table_dptr;
  libhu::UPTR hash_table_dptr;
  
  thrust::fill(hash_table_begin, hash_table_end, hf.PACKED_UNDEFINED_KEY);
  hash_table_dptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin)); 
  
  hf.hash_tableUPtr    = hash_table_dptr;
  hf.hash_table_size   = hash_table_size;
  
  libhu::U32 max_table_size = (!constrained_hash_access) ? hash_table_size : 1;
  thrust::device_vector<libhu::U32> max_table(max_table_size);
  thrust::fill(max_table.begin(), max_table.end(), hf.KEY_TYPE_NULL_AGE);  

  max_table_dptr   = (libhu::UPTR)(thrust::raw_pointer_cast(&*max_table.begin()));
  hf.max_tableUPtr = (libhu::U32*)max_table_dptr;  
  
  const unsigned int max1DBlocks = 65535;
  const unsigned int max2DBlocks = 65535u * 65535u;
  
  for (unsigned int block = 0; block < num_blocks; block += max2DBlocks)
  {
    unsigned int blocks          = min(max2DBlocks, num_blocks - block);
    unsigned int keys_size_range = min(blocks * block_size, keys_size - (block * block_size));
    
    libhu::UPTR keys_range_dptr   = (libhu::UPTR)(thrust::raw_pointer_cast(&*(keys_begin   + (block * block_size)) ));
    libhu::UPTR values_range_dptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*(values_begin + (block * block_size)) ));

    // Round up in order to make sure all items are hashed in.
    const unsigned grid_size  = 16384;
    dim3 grid( (keys_size_range +block_size-1) / block_size );
    if (grid.x > grid_size) {
        grid.y = (grid.x + grid_size - 1) / grid_size;
        grid.x = grid_size;
    }

#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
    NVPA_START_EVENT_GPU(cudakernelBuildHash);
    NVPA_GET_KERNEL_EVENT_MEMORY(num_blocks, block_size);
#endif

  if (constrained_hash_access)
  {
    libhu::constrained_hash_access_tag constrained_hash_access;
    hash_by_key_kernel
     <T_KEY, T_VALUE, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
     <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_VALUE*)values_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_MAX_AGE*)max_table_dptr, hf, constrained_hash_access);
  }
  else
  {
    libhu::unconstrained_hash_access_tag unconstrained_hash_access;
    hash_by_key_kernel
     <T_KEY, T_VALUE, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
     <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_VALUE*)values_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_MAX_AGE*)max_table_dptr, hf, unconstrained_hash_access);
  }
  thrust::detail::device::cuda::synchronize_if_enabled("hash_by_key_kernel");

#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
    NVPA_STOP_EVENT_GPU(cudakernelBuildHash);
#endif

    libhu::F32 build_time;

  }

  if (constrained_hash_access)
  {
    // Compute global max. age
    //T_RAND_ACCESS_ITERATOR2 max_iter;
    //T_HASH_TABLE            key_and_age;
    //max_iter = thrust::max_element(hash_table_begin, hash_table_end);
    //thrust::copy(hash_table_begin + (max_iter - hash_table_begin), hash_table_begin + ((max_iter - hash_table_begin) + 1), &key_and_age);
    //max_age = hf.GET_KEY_AGE(key_and_age) + 1;
    // Set limit max. age
    max_age = hf.KEY_TYPE_MAX_AGE;
  }
  else
  {
    // Add max. chain to the hash table
    thrust::for_each(hash_table_begin, hash_table_end, hf);
  }

}

// ------------------------------------------------------------------

} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace libh


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

