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

#include <iostream>
#include <libhu/detail/backend/cpp/detail/hash_utils.h>

namespace libhu
{
namespace detail
{
namespace backend
{
namespace cpp
{
namespace detail
{

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR,
         typename T_MAX_AGE_COMPUTATION_FUNCTOR>
__inline__ __host__
void access_kernel_host(libhu::U32                    search_keys_size,
                        libhu::U32                    hash_table_size,
                        T_KEY*                        search_keys,
                        T_HASH_TABLE*                 hash_table,
                        T_VALUE*                      output_values,
                        T_HASH_FUNCTOR                hf,
                        T_MAX_AGE_COMPUTATION_FUNCTOR maf,
                        libhu::U32                    max_age)
{

#if (!__CUDACC__)
  for (libhu::U32 GTID = 0; GTID < search_keys_size; GTID++)
#endif
  {

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
  }

}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR>
void access_kernel(libhu::U32                         search_keys_size,
                   libhu::U32                         hash_table_size,
                   T_KEY*                             search_keys,
                   T_HASH_TABLE*                      hash_table,
                   T_VALUE*                           output_values,
                   T_HASH_FUNCTOR                     hf,
                   libhu::constrained_hash_access_tag constrained_hash_access,
                   libhu::U32                         max_age)
{
  access_kernel_host(search_keys_size, hash_table_size, search_keys, hash_table, output_values, hf, constrained_hash_access, max_age);
}

// ------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_VALUE,
         typename T_HASH_FUNCTOR>
void access_kernel(libhu::U32                           search_keys_size,
                   libhu::U32                           hash_table_size,
                   T_KEY*                               search_keys,
                   T_HASH_TABLE*                        hash_table,
                   T_VALUE*                             output_values,
                   T_HASH_FUNCTOR                       hf,
                   libhu::unconstrained_hash_access_tag unconstrained_hash_access,
                   libhu::U32                           max_age)
{
  access_kernel_host(search_keys_size, hash_table_size, search_keys, hash_table, output_values, hf, unconstrained_hash_access, max_age);
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

              libhu::U32 keys_size            = search_keys_end - search_keys_begin;
              libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;
              
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_HASH_TABLE;
              typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR3>::type T_VALUE;

              libhu::UPTR search_keys_hptr   = (libhu::UPTR)(thrust::raw_pointer_cast(&*search_keys_begin));
              libhu::UPTR output_values_hptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*output_values_begin));
              libhu::UPTR hash_table_hptr    = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin));
              
              hf.hash_tableUPtr  = hash_table_hptr;
              hf.hash_table_size = hash_table_size;
              
              if (constrained_hash_access)
              {
                libhu::constrained_hash_access_tag constrained_hash_access;
                access_kernel
                 <T_KEY, T_HASH_TABLE, T_VALUE, T_HASH_FUNCTOR>
                 (keys_size, hash_table_size, (T_KEY*)search_keys_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_VALUE*)output_values_hptr, hf, constrained_hash_access, max_age);
              }
              else
              {
                libhu::unconstrained_hash_access_tag unconstrained_hash_access;
                access_kernel
                 <T_KEY, T_HASH_TABLE, T_VALUE, T_HASH_FUNCTOR>
                 (keys_size, hash_table_size, (T_KEY*)search_keys_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_VALUE*)output_values_hptr, hf, unconstrained_hash_access, max_age);
              }

            }

} // end namespace detail
} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace libhu

