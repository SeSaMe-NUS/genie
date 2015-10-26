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
#include <libh/detail/backend/cpp/detail/hash.h>

namespace libh
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
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
void hash_kernel(libhu::U32                         keys_size,
                 libhu::U32                         hash_table_size,
                 T_KEY*                             keys,
                 T_HASH_TABLE*                      hash_table,
                 T_MAX_AGE*                         max_table,
                 T_HASH_FUNCTOR                     hf,
                 libhu::constrained_hash_access_tag constrained_hash_access)
{
  hf.hash_kernel(keys_size, hash_table_size, keys, hash_table, max_table, hf);
}

//------------------------------------------------------------------------

template<typename T_KEY,
         typename T_HASH_TABLE,
         typename T_MAX_AGE,
         typename T_HASH_FUNCTOR>
void hash_kernel(libhu::U32                           keys_size,
                 libhu::U32                           hash_table_size,
                 T_KEY*                               keys,
                 T_HASH_TABLE*                        hash_table,
                 T_MAX_AGE*                           max_table,
                 T_HASH_FUNCTOR                       hf,
                 libhu::unconstrained_hash_access_tag unconstrained_hash_access)
{
  hf.hash_kernel(keys_size, hash_table_size, keys, hash_table, max_table, hf);
}

//------------------------------------------------------------------------

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

  libhu::U32 keys_size            = keys_end - keys_begin;
  libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;
       
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_HASH_TABLE;
  typedef typename thrust::host_vector<libhu::U32>::iterator T_RAND_ACCESS_ITERATOR4;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR4>::type T_MAX_AGE;

  thrust::fill(hash_table_begin, hash_table_end, hf.PACKED_UNDEFINED_KEY);
  
  hf.hash_tableUPtr    = hash_table_begin;
  hf.hash_table_size   = hash_table_size;
  
  libhu::U32 max_table_size = (!constrained_hash_access) ? hash_table_size : 1;
  thrust::host_vector<libhu::U32> max_table(max_table_size);
  thrust::fill(max_table.begin(), max_table.end(), hf.KEY_TYPE_NULL_AGE);  

  libhu::UPTR keys_hptr       = (libhu::UPTR)(thrust::raw_pointer_cast(&*keys_begin));
  libhu::UPTR hash_table_hptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin));
  libhu::UPTR max_table_hptr  = (libhu::UPTR)(thrust::raw_pointer_cast(&*max_table.begin()));

  hf.max_tableUPtr = (libhu::U32*)max_table_hptr;

  if (constrained_hash_access)
  {
    libhu::constrained_hash_access_tag constrained_hash_access;
    hash_kernel
      <T_KEY, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
      (keys_size, hash_table_size, (T_KEY*)keys_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_MAX_AGE*)max_table_hptr, hf, constrained_hash_access);
  }
  else
  {
    libhu::unconstrained_hash_access_tag unconstrained_hash_access;
    hash_kernel
      <T_KEY, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
      (keys_size, hash_table_size, (T_KEY*)keys_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_MAX_AGE*)max_table_hptr, hf, unconstrained_hash_access);
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

//------------------------------------------------------------------------

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

  //------------------------------------------------------------------------

  //ensureTargetISAIsSupported();

  //int Nx = 256, Ny = 256, Nz = 256;
  //int width = 4;
  //float *Aserial[2], *Aispc[2];
  //Aserial[0] = new float [Nx * Ny * Nz];
  //Aserial[1] = new float [Nx * Ny * Nz];
  //Aispc[0]   = new float [Nx * Ny * Nz];
  //Aispc[1]   = new float [Nx * Ny * Nz];
  //float *vsq = new float [Nx * Ny * Nz];

  //float coeff[4] = { 0.5, -.25, .125, -.0625 }; 

  //InitData(Nx, Ny, Nz, Aispc, vsq);

  ////
  //// Compute the image using the ispc implementation on one core; report
  //// the minimum time of three runs.
  ////
  //double minTimeISPC = 1e30;
  //for (int i = 0; i < 3; ++i) {
  //    reset_and_start_timer();
  //    loop_cohash_ispc(0, 6, width, Nx - width, width, Ny - width,
  //                      width, Nz - width, Nx, Ny, Nz, coeff, vsq,
  //                      Aispc[0], Aispc[1]);
  //    double dt = get_elapsed_mcycles();
  //    minTimeISPC = std::min<double>(minTimeISPC, dt);
  //}

  //printf("[cohash ispc 1 core]:\t\t[%.3f] million cycles\n", minTimeISPC);

  //InitData(Nx, Ny, Nz, Aispc, vsq);

  ////
  //// Compute the image using the ispc implementation with tasks; report
  //// the minimum time of three runs.
  ////
  //double minTimeISPCTasks = 1e30;
  //for (int i = 0; i < 3; ++i) {
  //    reset_and_start_timer();
  //    loop_cohash_ispc_tasks(0, 6, width, Nx - width, width, Ny - width,
  //                            width, Nz - width, Nx, Ny, Nz, coeff, vsq,
  //                            Aispc[0], Aispc[1]);
  //    double dt = get_elapsed_mcycles();
  //    minTimeISPCTasks = std::min<double>(minTimeISPCTasks, dt);
  //}

  //printf("[cohash ispc + tasks]:\t\t[%.3f] million cycles\n", minTimeISPCTasks);

  //InitData(Nx, Ny, Nz, Aserial, vsq);

  //// 
  //// And run the serial implementation 3 times, again reporting the
  //// minimum time.
  ////
  //double minTimeSerial = 1e30;
  //for (int i = 0; i < 3; ++i) {
  //    reset_and_start_timer();
  //    loop_cohash_serial(0, 6, width, Nx-width, width, Ny - width,
  //                        width, Nz - width, Nx, Ny, Nz, coeff, vsq,
  //                        Aserial[0], Aserial[1]);
  //    double dt = get_elapsed_mcycles();
  //    minTimeSerial = std::min<double>(minTimeSerial, dt);
  //}

  //printf("[cohash serial]:\t\t[%.3f] millon cycles\n", minTimeSerial);

  //printf("\t\t\t\t(%.2fx speedup from ISPC, %.2f from ISPC + tasks)\n", 
  //        minTimeSerial / minTimeISPC, minTimeSerial / minTimeISPCTasks);

  //// Check for agreement
  //int offset = 0;
  //for (int z = 0; z < Nz; ++z)
  //    for (int y = 0; y < Ny; ++y)
  //        for (int x = 0; x < Nx; ++x, ++offset) {
  //            float error = fabsf((Aserial[1][offset] - Aispc[1][offset]) /
  //                                Aserial[1][offset]);
  //            if (error > 1e-4)
  //                printf("Error @ (%d,%d,%d): ispc = %f, serial = %f\n",
  //                        x, y, z, Aispc[1][offset], Aserial[1][offset]);
  //        }
  
  //------------------------------------------------------------------------

  libhu::U32 keys_size            = keys_end - keys_begin;
  libhu::U32 hash_table_size      = hash_table_end - hash_table_begin;

  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR1>::type T_KEY;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR2>::type T_VALUE;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR3>::type T_HASH_TABLE;
  typedef typename thrust::host_vector<libhu::U32>::iterator T_RAND_ACCESS_ITERATOR4;
  typedef typename thrust::iterator_value<T_RAND_ACCESS_ITERATOR4>::type T_MAX_AGE;  

  thrust::fill(hash_table_begin, hash_table_end, hf.PACKED_UNDEFINED_KEY);

  hf.hash_tableUPtr    = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin));
  hf.hash_table_size   = hash_table_size;
  
  libhu::U32 max_table_size = (!constrained_hash_access) ? hash_table_size : 1;
  thrust::host_vector<libhu::U32> max_table(max_table_size);
  thrust::fill(max_table.begin(), max_table.end(), hf.KEY_TYPE_NULL_AGE);

  libhu::UPTR keys_hptr       = (libhu::UPTR)(thrust::raw_pointer_cast(&*keys_begin));
  libhu::UPTR values_hptr     = (libhu::UPTR)(thrust::raw_pointer_cast(&*values_begin));
  libhu::UPTR hash_table_hptr = (libhu::UPTR)(thrust::raw_pointer_cast(&*hash_table_begin));
  libhu::UPTR max_table_hptr  = (libhu::UPTR)(thrust::raw_pointer_cast(&*max_table.begin()));

  hf.max_tableUPtr = (libhu::U32*)max_table_hptr;

  if (constrained_hash_access)
  {
    libhu::constrained_hash_access_tag constrained_hash_access;
    hash_by_key_kernel
      <T_KEY, T_VALUE, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
      (keys_size, hash_table_size, (T_KEY*)keys_hptr, (T_VALUE*)values_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_MAX_AGE*)max_table_hptr, hf, constrained_hash_access);
  }
  else
  {
    libhu::unconstrained_hash_access_tag unconstrained_hash_access;
    hash_by_key_kernel
      <T_KEY, T_VALUE, T_HASH_TABLE, T_MAX_AGE, T_HASH_FUNCTOR>
      (keys_size, hash_table_size, (T_KEY*)keys_hptr, (T_VALUE*)values_hptr, (T_HASH_TABLE*)hash_table_hptr, (T_MAX_AGE*)max_table_hptr, hf, unconstrained_hash_access);
  }

  libhu::F32 build_time;

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

} // end namespace detail
} // end namespace cpp
} // end namespace backend
} // end namespace detail
} // end namespace libh

