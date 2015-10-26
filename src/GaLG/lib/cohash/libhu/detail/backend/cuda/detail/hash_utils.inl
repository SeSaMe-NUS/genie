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

namespace libhu
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR,
         typename T_MAX_AGE_COMPUTATION_FUNCTOR>
__inline__ __device__   
void access_kernel_device(libhu::U32                    search_keys_size,
                          libhu::U32                    hash_table_size,
                          T_KEY*                        search_keys,
                          T_HASH_TABLE*                 hash_table,
                          T_VALUE*                      output_values,
                          T_HASH_FUNCTOR                hf,
                          T_MAX_AGE_COMPUTATION_FUNCTOR maf,
                          libhu::U32                    max_age)
{

#if (ENABLE_NVPA)
#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
  NVPA_KERNEL_START_EVENT(cudainternalAccessHash);
#endif
#endif

  libhu::U32   LOC;
  T_HASH_TABLE OUT_KEY;  
  libhu::U8    AGE     = hf.KEY_TYPE_NULL_AGE;
  libhu::U8    MAX_AGE = max_age;
  T_KEY        KEY     = (GTID < search_keys_size) ? search_keys[ (GTID) ] : hf.PACKED_UNDEFINED_KEY;

  if (KEY != hf.PACKED_UNDEFINED_KEY)
  {
    LOC         = hf.h(KEY, AGE, hash_table_size);
    OUT_KEY     = hash_table[LOC];
    
    maf.get_max_age(OUT_KEY, max_age, MAX_AGE, hf);
    
    if (hf.GET_KEY_POS(OUT_KEY) == KEY)
    {
      output_values[ (GTID) ] = OUT_KEY;
      return;
    }
    
    while (AGE < MAX_AGE)
    {
      AGE++;
      LOC      = hf.h(KEY, AGE, hash_table_size);
      OUT_KEY  = hash_table[LOC];
   
      if (hf.GET_KEY_POS(OUT_KEY) == KEY)
      {
        //output_values[ (GTID) ] = hf.GET_KEY_ATTACH_ID(OUT_KEY);
        output_values[ (GTID) ] = hf.GET_KEY_POS(OUT_KEY);
        return;
      }
    }
    
    output_values[ (GTID) ] = hf.PACKED_UNDEFINED_KEY;
  }

#if (ENABLE_NVPA)
#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
  NVPA_KERNEL_STOP_EVENT(cudainternalAccessHash);
#endif
#endif

}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR>
__global__ 
void access_kernel(libhu::U32                         search_keys_size,
                   libhu::U32                         hash_table_size,
                   T_KEY*                             search_keys,
                   T_HASH_TABLE*                      hash_table,
                   T_VALUE*                           output_values,
                   T_HASH_FUNCTOR                     hf,
                   libhu::constrained_hash_access_tag constrained_hash_access,
                   libhu::U32                         max_age)
{
  access_kernel_device(search_keys_size, hash_table_size, search_keys, hash_table, output_values, hf, constrained_hash_access, max_age);
}

// ------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR>
__global__ 
void access_kernel(libhu::U32                           search_keys_size,
                   libhu::U32                           hash_table_size,
                   T_KEY*                               search_keys,
                   T_HASH_TABLE*                        hash_table,
                   T_VALUE*                             output_values,
                   T_HASH_FUNCTOR                       hf,
                   libhu::unconstrained_hash_access_tag unconstrained_hash_access,
                   libhu::U32                           max_age)
{
  access_kernel_device(search_keys_size, hash_table_size, search_keys, hash_table, output_values, hf, unconstrained_hash_access, max_age);
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
              T_HASH_FUNCTOR          hf,
              bool                    constrained_hash_access,
              libhu::U32              max_age)
            {
            
              const libhu::U32 block_size     = hf.DEFAULT_GROUP_SIZE;
              libhu::U32 keys_size            = search_keys_end - search_keys_begin;
              libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;
              libhu::U32 num_blocks           = ceilf((libhu::F32)keys_size / (libhu::F32)block_size);
              libhu::U32 smem_size            = 0u;                
              
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_HASH_TABLE;
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR3>::type T_VALUE;
                
              libhu::UPTR hash_table_dptr  = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin)); 
              
              hf.hash_tableUPtr    = hash_table_dptr;
              hf.hash_table_size   = hash_table_size;
              
              const unsigned int max1DBlocks = 65535u;
              const unsigned int max2DBlocks = 65535u * 65535u;

              for (unsigned int block = 0; block < num_blocks; block += max2DBlocks)
              {
                unsigned int blocks          = min(max2DBlocks, num_blocks - block);
                unsigned int keys_size_range = min(blocks * block_size, keys_size - (block * block_size));
                
                libhu::UPTR keys_range_dptr          = (libhu::UPTR)(thrust::raw_pointer_cast(&*(search_keys_begin   + (block * block_size)) ));
                libhu::UPTR output_values_range_dptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*(output_values_begin   + (block * block_size)) ));

                // Round up in order to make sure all items are hashed in.
                const unsigned grid_size  = 16384;
                dim3 grid( (keys_size_range +block_size-1) / block_size );
                if (grid.x > grid_size) {
                    grid.y = (grid.x + grid_size - 1) / grid_size;
                    grid.x = grid_size;
                }

#if (ENABLE_NVPA)
#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
                NVPA_START_EVENT_GPU(cudakernelAccessHash);
                NVPA_GET_KERNEL_EVENT_MEMORY(grid.x * grid.y, block_size);
#endif
#endif

                if (constrained_hash_access)
                {
                  libhu::constrained_hash_access_tag constrained_hash_access;
                  access_kernel
                   <T_KEY, T_HASH_TABLE, T_VALUE, T_HASH_FUNCTOR>
                   <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_VALUE*)output_values_range_dptr, hf, constrained_hash_access, max_age);
                }
                else
                {
                  libhu::unconstrained_hash_access_tag unconstrained_hash_access;
                  access_kernel
                   <T_KEY, T_HASH_TABLE, T_VALUE, T_HASH_FUNCTOR>
                   <<<grid, block_size, smem_size>>>(keys_size_range, hash_table_size, (T_KEY*)keys_range_dptr, (T_HASH_TABLE*)hash_table_dptr, (T_VALUE*)output_values_range_dptr, hf, unconstrained_hash_access, max_age);
                }
                thrust::detail::device::cuda::synchronize_if_enabled("access_kernel");
#if (ENABLE_NVPA)
#if (!DISABLE_NVPA_KERNEL_EVEN_MEMORY_DECLARATION)
                NVPA_STOP_EVENT_GPU(cudakernelAccessHash);
#endif
#endif

              } 
            }

// ------------------------------------------------------------------

} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace libhu


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

