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

#include <thrust/copy.h>
#include <thrust/detail/trivial_sequence.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>

#include <libhu/detail/backend/generic/hash_utils.h>
#include <libhu/detail/backend/cpp/hash_utils.h>
#include <libhu/detail/backend/cuda/hash_utils.h>

#include <libhu/hash_utils.h>

namespace libhu
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template<typename T_RAND_ACCESS_ITERATOR1,
         typename T_RAND_ACCESS_ITERATOR2,
         typename T_RAND_ACCESS_ITERATOR3,
         typename T_HASH_FUNCTOR,
         typename Backend>
  void access(T_RAND_ACCESS_ITERATOR1 search_keys_begin,
              T_RAND_ACCESS_ITERATOR1 search_keys_end,
              T_RAND_ACCESS_ITERATOR2 hash_table_begin,
              T_RAND_ACCESS_ITERATOR2 hash_table_end,
              T_RAND_ACCESS_ITERATOR3 output_values_begin,
              T_HASH_FUNCTOR          hf,
              bool                    constrained_hash_access,
              libhu::U32              max_age,
              Backend)
{
  libhu::detail::backend::generic::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
}

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
              libhu::U32              max_age,
              thrust::host_space_tag)
{
  //if (checkTargetISAIsSupported())
  //{
  //  std::cerr << "libhu::detail::backend::ispc::access(...)" << std::endl;
  //  libhu::detail::backend::ispc::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
  //}
  //else
  {
    std::cerr << "libhu::detail::backend::cpp::access(...)" << std::endl;
    libhu::detail::backend::cpp::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age);
  }
}

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
              libhu::U32              max_age,
              thrust::detail::cuda_device_space_tag)
{
  // ensure sequence has trivial iterators
  // XXX this prologue belongs somewhere else
  thrust::detail::trivial_sequence<T_RAND_ACCESS_ITERATOR1> keys(search_keys_begin, search_keys_end);
  thrust::detail::trivial_sequence<T_RAND_ACCESS_ITERATOR2> hash_table(hash_table_begin, hash_table_end);

  libhu::detail::backend::cuda::access(keys.begin(), keys.end(), hash_table.begin(), hash_table.end(), output_values_begin, hf, constrained_hash_access, max_age);

  // copy results back, if necessary
  // XXX this epilogue belongs somewhere else
  if(!thrust::detail::is_trivial_iterator<T_RAND_ACCESS_ITERATOR1>::value)
    thrust::copy(hash_table.begin(), hash_table.end(), hash_table_begin);

}

} // end namespace dispatch

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
  libhu::detail::backend::dispatch::access(search_keys_begin, search_keys_end, hash_table_begin, hash_table_end, output_values_begin, hf, constrained_hash_access, max_age,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<T_RAND_ACCESS_ITERATOR1>::type,
      typename thrust::iterator_space<T_RAND_ACCESS_ITERATOR1>::type,
      typename thrust::iterator_space<T_RAND_ACCESS_ITERATOR2>::type,
      typename thrust::iterator_space<T_RAND_ACCESS_ITERATOR2>::type,
      typename thrust::iterator_space<T_RAND_ACCESS_ITERATOR3>::type
      >::type());

}

} // end namespace backend
} // end namespace detail
} // end namespace libhu

