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

template <typename T>
struct copy_tex_value
{
  static const libhu::U32 KEY_TYPE_BITS                 = 32u;
  static const libhu::U32 KEY_TYPE_MASK                 = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1ul );
  static const libhu::U32 ATTACH_ID_TYPE_BITS           = 28u;
  static const libhu::U32 ATTACH_ID_TYPE_MASK           = libhu::U32( libhu::U64((1ull) << ATTACH_ID_TYPE_BITS) - 1ul );
  static const libhu::U32 PACKED_UNDEFINED_KEY          = libhu::U32( libhu::U64((1ull) << KEY_TYPE_BITS) - 1ul );

  inline __host__ __device__
    copy_tex_value(libhu::UPTR _keysUPtr, libhu::UPTR _valuesUPtr, libhu::UPTR _refValuesUPtr, libhu::U32 _W, libhu::U32 _H) : keysUPtr(_keysUPtr), valuesUPtr(_valuesUPtr), refValuesUPtr(_refValuesUPtr), W(_W), H(_H) {} 

  inline __host__ __device__ __forceinline__ libhu::U32 GET_KEY_POS(libhu::U64 k) { return ((k) & KEY_TYPE_MASK); }
  inline __host__ __device__ __forceinline__ libhu::U32 GET_KEY_ATTACH_ID(libhu::U64 k) { return ((k) >> (KEY_TYPE_BITS)) & ATTACH_ID_TYPE_MASK; }

  inline __host__ __device__
  void operator()(T& t)
  {
    libhu::U32* valuesPtr = (libhu::U32*)valuesUPtr;
    libhu::U32* refValuesPtr = (libhu::U32*)refValuesUPtr;
    if (t != PACKED_UNDEFINED_KEY)
    {
      //valuesPtr[GET_KEY_POS(t)] = GET_KEY_ATTACH_ID(GET_KEY_POS(t));
      valuesPtr[GET_KEY_POS(t)] = refValuesPtr[GET_KEY_POS(t)];
    }
  }

  libhu::UPTR keysUPtr;
  libhu::UPTR valuesUPtr;
  libhu::UPTR refValuesUPtr;
  libhu::U32 W;
  libhu::U32 H;
};

int testRobinHoodHash(ConfigParams& cfg)
{
  libhu::U32 num_keys     = cfg.num_keys;
  libhu::U32 num_extra    = cfg.num_extra;
  libhu::U32 num_queries  = (cfg.rate_non_valid_keys == 0) ? cfg.num_keys : cfg.num_extra;
  libhu::F32 dens         = cfg.dens;
  libhu::U32 seed         = cfg.seed;
  bool sorted_access      = cfg.sorted_access;
      
  cudaEvent_t start  = 0u;
  cudaEvent_t end    = 0u;
  libhu::F32 coh_build_time;
  libhu::F32 coh_access_time;
  libhu::F32 rand_build_time;
  libhu::F32 rand_access_time;
  
  CUdevice     device;
  bool cvsOutputMode = true;
  
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  libhu::U32 hash_table_size = ceilf(num_keys / dens);
  hash_table_size = ((hash_table_size & 1u) == 0u) ? hash_table_size + 1u : hash_table_size;
  
  std::cerr << "mode                             : " << ((ENABLE_KEY_DATA_MODE == 1) ? "key_value (64b)" : "key (32b)") << std::endl;
  std::cerr << "num_keys                         : " << num_keys << std::endl;
  std::cerr << "num_extra                        : " << num_extra << std::endl;
  std::cerr << "num_queries                      : " << num_queries << std::endl;
  std::cerr << "rate_non_valid_keys              : " << cfg.rate_non_valid_keys << std::endl;
  std::cerr << "hash_table_size                  : " << hash_table_size << std::endl;  
  std::cerr << "density                          : " << dens << std::endl;
  std::cerr << "seed                             : " << seed << std::endl;
  std::cerr << "sorted                           : " << sorted_access << std::endl;
  
  typedef libhu::U32 T_KEY;
  typedef libhu::U32 T_VALUE;
  typedef thrust::device_vector<T_KEY>::iterator T_KEY_IT;
  typedef thrust::device_vector<T_VALUE>::iterator T_VALUE_IT;

#if (!ENABLE_KEY_DATA_MODE)
  typedef libhu::U32 T_HASH_TABLE;
  typedef thrust::device_vector<T_HASH_TABLE>::iterator T_HASH_TABLE_IT;
  typedef key_coh_hash_functor  T_COH_FUNCTOR;
  typedef key_rand_hash_functor T_RAND_FUNCTOR;
  key_rand_hash_functor   krf = key_rand_hash_functor();
  key_coh_hash_functor    kcf = key_coh_hash_functor();
  libhu::U32    key_max_value = krf.KEY_TYPE_RANGE;
  libhu::U32      key_max_age = krf.KEY_TYPE_MAX_AGE;
  thrust::device_vector<T_KEY>        values(num_extra);
  thrust::device_vector<T_HASH_TABLE> hash_table(hash_table_size);
#elif (ENABLE_KEY_DATA_MODE)
  typedef libhu::U64 T_HASH_TABLE;
  typedef thrust::device_vector<T_HASH_TABLE>::iterator T_HASH_TABLE_IT;
  typedef key_value_coh_hash_functor  T_COH_FUNCTOR;
  typedef key_value_rand_hash_functor T_RAND_FUNCTOR;
  key_value_rand_hash_functor krf = key_value_rand_hash_functor();
  key_value_coh_hash_functor  kcf = key_value_coh_hash_functor();
  libhu::U32        key_max_value = krf.KEY_TYPE_RANGE;
  libhu::U32          key_max_age = krf.KEY_TYPE_MAX_AGE;
  thrust::device_vector<T_VALUE>      values(num_extra);
  thrust::device_vector<T_HASH_TABLE> hash_table(hash_table_size);
#endif

  thrust::device_vector<T_KEY> keys(num_extra);
  thrust::fill(keys.begin(), keys.end(), kcf.UNDEFINED_KEY);

  if (cfg.rand_num_mode)
  {

#if (ENABLE_LIBHU_GPU_RANDOM_GENERATOR)
  
#if (ENABLE_2D_COORDS)
    libhu::key_2d<libhu::U32, kcf.KEY_TYPE_BITS> key_2d_converter;
    key_max_value = (key_2d_converter.w * key_2d_converter.h) - 1u;
    //libhu::key_2d<libhu::U32, kcf.KEY_TYPE_BITS> key_2d_converter(cfg.u2D_w, cfg.u2D_h);
    //key_max_value = (key_2d_converter.w * key_2d_converter.h) - 1u;
    if (key_max_value < cfg.num_extra) 
    {
      std::cerr << "Error: num_extra > universe size" << std::endl;
      exit(0);
    }
    bool unique = libhu::generate_unique_random(keys.begin(), keys.end(), 1u, key_max_value, seed);
    std::cerr << "2D universe (" << key_2d_converter.w << "," << key_2d_converter.h << ")" << std::endl;
#elif (ENABLE_3D_COORDS)
    libhu::key_3d<libhu::U32, kcf.KEY_TYPE_BITS> key_3d_converter;  
    key_max_value = (key_3d_converter.w * key_3d_converter.h * key_3d_converter.d) - 1u;
    //libhu::key_3d<libhu::U32, kcf.KEY_TYPE_BITS> key_3d_converter(cfg.u3D_w, cfg.u3D_h, cfg.u3D_d);
    //key_max_value = (key_3d_converter.w * key_3d_converter.h * key_3d_converter.d) - 1u;  
    if (key_max_value < cfg.num_extra) 
    {
      std::cerr << "Error: num_extra > universe size" << std::endl;
      exit(0);
    }
    bool unique = libhu::generate_unique_random(keys.begin(), keys.end(), 1u, key_max_value, seed);
    std::cerr << "3D universe (" << key_3d_converter.w << "," << key_3d_converter.h << "," << key_3d_converter.d << ")" << std::endl;
#endif

#else
    unsigned int *h_keysInsert   = new T_KEY  [num_extra];
    VectorSupport<unsigned int>::fillVectorForHash(h_keysInsert, 0.0, 1.0, false, false, cfg.num_extra, cfg.seed);
    thrust::copy(h_keysInsert, h_keysInsert + num_extra, keys.begin());
    thrust::copy(keys.begin(), keys.end(), values.begin());
    delete [] h_keysInsert;
#endif

    if (sorted_access)
    {
      thrust::sort(keys.begin(), keys.end());
    }
    else
    {
      thrust::host_vector<T_KEY> host_keys(num_extra); 
      thrust::copy(keys.begin(), keys.end(), host_keys.begin());
      srand(cfg.seed);
      std::random_shuffle(host_keys.begin(), host_keys.end());
      thrust::copy(host_keys.begin(), host_keys.end(), keys.begin()); 
    }
    thrust::copy(keys.begin(), keys.end(), values.begin());

  }
  else
  {
    libhu::U32 tnnz = 0;
    thrust::host_vector<libhu::U32> host_keys(cfg.num_extra);

    for (libhu::U32 i = 0; i < cfg.tex->w * cfg.tex->h; i++)
    {
      libhu::U32 *imgPtr = (libhu::U32*)cfg.tex->data;
      if (imgPtr[i] != 0)
      {
      host_keys[tnnz] = i;
      tnnz++;
      }
    }
    
    if (cfg.num_keys != cfg.num_extra)
    {    
      for (libhu::U32 i = 0; i < cfg.tex->w * cfg.tex->h; i++)
      {
        libhu::U32 *imgPtr = (libhu::U32*)cfg.tex->data;
        if (imgPtr[i] == 0)
        {
          host_keys[tnnz] = i;
          tnnz++;
        }
      }
    }
    
    if (!cfg.sorted_access)
    {
      std::random_shuffle(host_keys.begin(), host_keys.begin() + cfg.num_keys);
      
      if (cfg.num_keys != cfg.num_extra)
      {
        std::random_shuffle(host_keys.begin() + cfg.num_keys, host_keys.begin() + cfg.num_extra);
      }
    }
    
    thrust::copy(host_keys.begin(), host_keys.end(), keys.begin());

#if (ENABLE_KEY_DATA_MODE)
    thrust::copy(host_keys.begin(), host_keys.end(), values.begin());
#endif

  }
  
  //libhu::print("search_keys", keys.begin(), keys.begin() + 4);

#if (ENABLE_KERNEL_PROFILING)
  CUresult resProf      = cuProfilerInitialize("profiling.cfg", "profiling.csv", (CUoutput_mode)((cvsOutputMode) ? CU_OUT_CSV : CU_OUT_KEY_VALUE_PAIR));
  CUresult resProfStart = cuProfilerStart();
#endif
  
  if (cfg.coh_hash)
  {
    {
      cudaEventRecord(start,0);

    #if (!ENABLE_KEY_DATA_MODE)
      libh::hash(keys.begin(), keys.begin() + num_keys, hash_table.begin(), hash_table.end(), kcf, (cfg.rate_non_valid_keys == 0), key_max_age);
    #elif (ENABLE_KEY_DATA_MODE)
      libh::hash_by_key(keys.begin(), keys.begin() + num_keys, values.begin(), hash_table.begin(), hash_table.end(), kcf, (cfg.rate_non_valid_keys == 0), key_max_age);
    #endif  
      cudaEventRecord(end,0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&coh_build_time, start, end);

      bool okCoh = check_hashed(hash_table.begin(), hash_table.end(), num_keys, kcf);
      std::cerr << "build coh_hash state             : " << okCoh << std::endl;

      cfg.rh_coh_hash_state = okCoh;
      cfg.rh_coh_hash_build_time = coh_build_time;

    }
     
    {

      if (!cfg.sorted_access)
      {
        thrust::host_vector<unsigned int> host_queries(num_queries);
        thrust::copy(keys.begin(), keys.begin() + num_queries, host_queries.begin());
        std::random_shuffle(host_queries.begin(), host_queries.begin() + num_queries);
        thrust::copy(host_queries.begin(), host_queries.begin() + num_queries, keys.begin());
      }
      else
      {
        thrust::sort(keys.begin(), keys.begin() + num_queries);
      }

      //libhu::print("query_keys", keys.begin(), keys.begin() + 4);

      cudaEventRecord(start,0);
      libhu::access(keys.begin(), keys.begin() + num_queries, hash_table.begin(), hash_table.end(), values.begin(), kcf, (cfg.rate_non_valid_keys == 0), key_max_age);
      cudaEventRecord(end,0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&coh_access_time, start, end);

      bool okCoh = true;
      //okCoh = libhu::check_access<T_KEY_IT, T_HASH_TABLE_IT, kcf.KEY_TYPE_MASK>(keys.begin(), keys.end(), values.begin());
      //std::cerr << "access coh:     " << okCoh << " - " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << coh_access_time << " milliseconds" << std::endl;
      
      cfg.rh_coh_hash_access_time = coh_access_time;

      //libhu::print("search_keys", keys.begin(), keys.end());

#if (ENABLE_KEY_DATA_MODE)
      if (cfg.image_mode)
      {

        thrust::host_vector<libhu::U32> host_values(cfg.tex->w * cfg.tex->h);
        thrust::device_vector<libhu::U32> dev_values(cfg.tex->w * cfg.tex->h);
        thrust::device_vector<libhu::U32> dev_refValues(cfg.tex->w * cfg.tex->h);

        thrust::copy(((libhu::U32*)cfg.tex->data), ((libhu::U32*)cfg.tex->data) + (cfg.tex->w * cfg.tex->h), dev_refValues.begin());
        thrust::for_each(values.begin(), values.end(), copy_tex_value<libhu::U32>((libhu::UPTR)thrust::raw_pointer_cast(&values[0]), (libhu::UPTR)thrust::raw_pointer_cast(&dev_values[0]), (libhu::UPTR)thrust::raw_pointer_cast(&dev_refValues[0]), cfg.tex->w, cfg.tex->h));
        thrust::copy(dev_values.begin(), dev_values.end(), host_values.begin());

        Texture* otexture = (Texture *)malloc(sizeof(Texture));
        otexture->w = cfg.tex->w;
        otexture->h = cfg.tex->h;
        otexture->depth = cfg.tex->depth;
        otexture->data = (unsigned char *)malloc(otexture->w * otexture->h * 4);
        thrust::copy(dev_values.begin(), dev_values.end(), (libhu::U32*)otexture->data);
        saveTGA(otexture, "hashed_image.tga");
        delete otexture;

      }
#endif

    }

    libhu::F32 TIME_1K_MILLISECONDS = 1000;
    libhu::F32 NUM_1M_KEYS = 1000000;
    libhu::F32 build_keys  = cfg.num_keys / NUM_1M_KEYS;
    libhu::F32 access_keys = ((cfg.rate_non_valid_keys == 0) ? cfg.num_keys : cfg.num_extra) / NUM_1M_KEYS;

    cfg.rh_coh_hash_build_keys_per_sec = ((cfg.rh_coh_hash_state) ? ((build_keys * TIME_1K_MILLISECONDS) / cfg.rh_coh_hash_build_time) : -1);
    cfg.rh_coh_hash_access_keys_per_sec = ((cfg.rh_coh_hash_state) ? ((access_keys * TIME_1K_MILLISECONDS) / cfg.rh_coh_hash_access_time) : -1);

  }
  else if (cfg.rand_hash)
  {

    {
      cudaEventRecord(start,0);

    #if (!ENABLE_KEY_DATA_MODE)
      libh::hash(keys.begin(), keys.begin() + num_keys, hash_table.begin(), hash_table.end(), krf, (cfg.rate_non_valid_keys == 0), key_max_age);
    #elif (ENABLE_KEY_DATA_MODE)
      libh::hash_by_key(keys.begin(), keys.begin() + num_keys, values.begin(), hash_table.begin(), hash_table.end(), krf, (cfg.rate_non_valid_keys == 0), key_max_age);
    #endif

      cudaEventRecord(end,0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&rand_build_time, start, end); 

      bool okRand = check_hashed(hash_table.begin(), hash_table.end(), num_keys, krf);
      std::cerr << "build rand_hash state            : " << okRand <<  std::endl;

      cfg.rh_rand_hash_state = okRand;
      cfg.rh_rand_hash_build_time = rand_build_time;

    }

    {

      if (!cfg.sorted_access)
      {
        thrust::host_vector<unsigned int> host_queries(num_queries);
        thrust::copy(keys.begin(), keys.begin() + num_queries, host_queries.begin());
        std::random_shuffle(host_queries.begin(), host_queries.begin() + num_queries);
        thrust::copy(host_queries.begin(), host_queries.begin() + num_queries, keys.begin());
      }
      else
      {
        thrust::sort(keys.begin(), keys.begin() + num_queries);
      }

      //libhu::print("query_keys", keys.begin(), keys.begin() + 4);

      cudaEventRecord(start,0);

      libhu::access(keys.begin(), keys.begin() + num_queries, hash_table.begin(), hash_table.end(), values.begin(), krf, (cfg.rate_non_valid_keys == 0), key_max_age);

      cudaEventRecord(end,0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&rand_access_time, start, end);

      bool okRand = true;
      //okRand = libhu::check_access<T_KEY_IT, T_HASH_TABLE_IT, krf.KEY_TYPE_MASK>(keys.begin(), keys.end(), values.begin());
      //std::cerr << "access rand:    " << okRand << " - " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << rand_access_time << " milliseconds" << std::endl;

      cfg.rh_rand_hash_access_time = rand_access_time;
      
      //libhu::print("search_keys", keys.begin(), keys.end());

#if (ENABLE_KEY_DATA_MODE)
      if (cfg.image_mode)
      {

        thrust::host_vector<libhu::U32> host_values(cfg.tex->w * cfg.tex->h);
        thrust::device_vector<libhu::U32> dev_values(cfg.tex->w * cfg.tex->h);
        thrust::device_vector<libhu::U32> dev_refValues(cfg.tex->w * cfg.tex->h);

        thrust::copy(((libhu::U32*)cfg.tex->data), ((libhu::U32*)cfg.tex->data) + (cfg.tex->w * cfg.tex->h), dev_refValues.begin());
        thrust::for_each(values.begin(), values.end(), copy_tex_value<libhu::U32>((libhu::UPTR)thrust::raw_pointer_cast(&values[0]), (libhu::UPTR)thrust::raw_pointer_cast(&dev_values[0]), (libhu::UPTR)thrust::raw_pointer_cast(&dev_refValues[0]), cfg.tex->w, cfg.tex->h));
        thrust::copy(dev_values.begin(), dev_values.end(), host_values.begin());

        Texture* otexture = (Texture *)malloc(sizeof(Texture));
        otexture->w = cfg.tex->w;
        otexture->h = cfg.tex->h;
        otexture->depth = cfg.tex->depth;
        otexture->data = (unsigned char *)malloc(otexture->w * otexture->h * 4);
        thrust::copy(dev_values.begin(), dev_values.end(), (libhu::U32*)otexture->data);
        saveTGA(otexture, "hashed_image.tga");
        delete otexture;

      }
#endif

    }
    
    std::cerr << "# hasked keys                    : " << num_keys << std::endl;
    std::cerr << "# accessed keys                  : " << num_queries << std::endl;

    libhu::F32 TIME_1K_MILLISECONDS = 1000;
    libhu::F32 NUM_1M_KEYS = 1000000;
    libhu::F32 build_keys  = cfg.num_keys / NUM_1M_KEYS;
    libhu::F32 access_keys = ((cfg.rate_non_valid_keys == 0) ? cfg.num_keys : cfg.num_extra) / NUM_1M_KEYS;

    cfg.rh_rand_hash_build_keys_per_sec = ((cfg.rh_rand_hash_state) ? ((build_keys * TIME_1K_MILLISECONDS) / cfg.rh_rand_hash_build_time) : -1);
    cfg.rh_rand_hash_access_keys_per_sec = ((cfg.rh_rand_hash_state) ? ((access_keys * TIME_1K_MILLISECONDS) / cfg.rh_rand_hash_access_time) : -1);

  }

#if (ENABLE_KERNEL_PROFILING)
  CUresult resProfStop = cuProfilerStop();
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
  
}