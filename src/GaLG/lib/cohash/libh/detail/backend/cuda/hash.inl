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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <thrust/gather.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>

#include <libh/detail/backend/cuda/detail/hash.h>

namespace libh
{
namespace detail
{
namespace backend
{
namespace cuda
{

namespace second_dispatch
{
    // second level of the dispatch decision tree

    template<typename T_RAND_ACCESS_ITERATOR1,
             typename T_RAND_ACCESS_ITERATOR2,
             typename T_HASH_FUNCTOR>
      void hash(T_RAND_ACCESS_ITERATOR1 keys_begin,
                T_RAND_ACCESS_ITERATOR1 keys_end,
                T_RAND_ACCESS_ITERATOR2 hash_table_begin,
                T_RAND_ACCESS_ITERATOR2 hash_table_end,
                T_HASH_FUNCTOR          hf,
                bool                    constrained_hash_access,
                libhu::U32             &max_age,
                thrust::detail::true_type)
    {
    
      libh::detail::backend::cuda::detail::hash(keys_begin, keys_end, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age);
      
    }
    
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
                     libhu::U32             &max_age,
                     thrust::detail::true_type)
    {
      libh::detail::backend::cuda::detail::hash_by_key(keys_begin, keys_end, values_begin, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age);
    }
    

} // end namespace second_dispatch


namespace first_dispatch
{
    // first level of the dispatch decision tree

    template<typename T_RAND_ACCESS_ITERATOR1,  
             typename T_RAND_ACCESS_ITERATOR2,  
             typename T_HASH_FUNCTOR>
      void hash(T_RAND_ACCESS_ITERATOR1 keys_begin,
                T_RAND_ACCESS_ITERATOR1 keys_end,
                T_RAND_ACCESS_ITERATOR2 hash_table_begin,
                T_RAND_ACCESS_ITERATOR2 hash_table_end,
                T_HASH_FUNCTOR          hf,
                bool                    constrained_hash_access,
                libhu::U32             &max_age,
                thrust::detail::true_type)
    {

      // decide whether to sort keys indirectly
      typedef typename thrust::iterator_traits<T_RAND_ACCESS_ITERATOR1>::value_type KeyType;
      static const bool hash_keys_indirectly = sizeof(KeyType) == 4;  

      // XXX WAR nvcc 3.0 unused variable warning
      (void) hash_keys_indirectly;
      
      second_dispatch::hash(keys_begin, keys_end, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age,
        thrust::detail::integral_constant<bool, hash_keys_indirectly>());
      
    }
    
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
                       libhu::U32             &max_age,
                       thrust::detail::true_type)
    {
    
      // decide whether to sort keys indirectly
      typedef typename thrust::iterator_traits<T_RAND_ACCESS_ITERATOR1>::value_type KeyType;
      static const bool hash_keys_indirectly = sizeof(KeyType) == 4;  

      // XXX WAR nvcc 3.0 unused variable warning
      (void) hash_keys_indirectly;
      
      second_dispatch::hash_by_key(keys_begin, keys_end, values_begin, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age,
        thrust::detail::integral_constant<bool, hash_keys_indirectly>());
        
    }

} // end namespace first_dispatch


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

  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<T_RAND_ACCESS_ITERATOR1, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );
  
  // dispatch on whether we can use radix_sort_by_key
  typedef typename thrust::iterator_traits<T_RAND_ACCESS_ITERATOR1>::value_type KeyType;
  static const bool use_hash_key_type = thrust::detail::is_arithmetic<KeyType>::value;

  // XXX WAR nvcc 3.0 unused variable warning
  (void) use_hash_key_type;
  
  first_dispatch::hash(keys_begin, keys_end, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age,
    thrust::detail::integral_constant<bool, use_hash_key_type>());
    
}

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

  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<T_RAND_ACCESS_ITERATOR1, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );
  
  // dispatch on whether we can use radix_sort_by_key
  typedef typename thrust::iterator_traits<T_RAND_ACCESS_ITERATOR1>::value_type KeyType;
  static const bool use_hash_key_type = thrust::detail::is_arithmetic<KeyType>::value;

  // XXX WAR nvcc 3.0 unused variable warning
  (void) use_hash_key_type;
  
  first_dispatch::hash_by_key(keys_begin, keys_end, values_begin, hash_table_begin, hash_table_end, hf, constrained_hash_access, max_age,
    thrust::detail::integral_constant<bool, use_hash_key_type>());

}

} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace libh

