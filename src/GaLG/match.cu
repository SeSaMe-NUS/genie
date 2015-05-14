#include "match.h"
#include <cmath>
#include <cstdlib.h>

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
#endif

#define OFFSETS_TABLE_16 {0u,3949349u,8984219u,9805709u,7732727u,1046459u,9883879u,4889399u,2914183u,3503623u,1734349u,8860463u,1326319u,1613597u,8604269u,9647369u}

#define NULL_AGE 0

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

namespace GaLG
{
  namespace device
  {
     const u32 DEFAULT_GROUP_SIZE            = 192u;
    
     const u32 KEY_TYPE_BITS                 = 28u;
     const u32 KEY_TYPE_MASK                 = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 PACKED_KEY_TYPE_MASK          = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 KEY_TYPE_RANGE                = u32( u64((1ull) << KEY_TYPE_BITS) - 2u );
     const u32 UNDEFINED_KEY                 = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 PACKED_UNDEFINED_KEY          = u32( u64((1ull) << KEY_TYPE_BITS) - 1ul);
    
     const u32 ATTACH_ID_TYPE_BITS           = 28u;
     const u32 ATTACH_ID_TYPE_MASK           = u32( u64((1ull) << ATTACH_ID_TYPE_BITS) - 1ul );
     const u32 UNDEFINED_ATTACH_ID           = u32( u64((1ull) << ATTACH_ID_TYPE_BITS) - 1ul );
     const u32 MAX_ATTACH_ID_TYPE            = u32( u64((1ull) << ATTACH_ID_TYPE_BITS) - 2ul );
    
    const u32 KEY_TYPE_AGE_MASK        = 15u;
    const u32 KEY_TYPE_AGE_BITS        = 4u;
    const u32 KEY_TYPE_INIT_AGE        = 1u;
    const u32 KEY_TYPE_NULL_AGE        = 0u;
    const u32 KEY_TYPE_MAX_AGE         = 16u;
    const u32 KEY_TYPE_MAX_AGE_MASK    = 4u;
    const u32 KEY_TYPE_MAX_AGE_BITS    = 4u;
    
    CUDA_CONST u32 offsets[16] = OFFSETS_TABLE_16;
    
    __inline__ __host__ __device__
    u32
    get_key_pos(u32 key)
    {
      return key & KEY_TYPE_MASK;
    }
    
    __inline__ __host__ __device__
    u32
    get_key_age(u32 key)
    {
      return key >> KEY_TYPE_BITS;
    }
    
    __inline__ __host__ __device__
    u32
    pack_key_pos_and_age(u32 key, u32 age)
    {
      return ((a << KEY_TYPE_BITS)+(key & KEY_TYPE_MASK));
    }
    
    __inline__ __host__ __device__
    u32
    hash(u32 key, u8 age, int hash_table_size){
      return (offsets[age] + key) % hash_table_size;
    }
    
    
    __inline__ __device__
    void
    access_kernel(u32 id,
                  u32* htable,
                  int table_size,
                  u32 * key_location,
                  int * key_found,
                  u32 max_age)
    {
      u32 location;
      u32 out_key;
      u8 age = KEY_TYPE_NULL_AGE;
      
      location = hash(id, age, hash_table_size);
      out_key = htable[location];
      
      if(out_key == id){
        * key_found = 1;
        * key_location = location;
        return;
      }
      
      //Key at root location is packed with its max age
      // in its hashing sequence.
      max_age = get_key_age(out_key);
      
      //Loop until max_age
      while(age < max_age){
        age ++;
        location = hash(id, age, hash_table_size);
        out_key = htable[location];
        
        if(get_key_pos(out_key) == id){
          * key_found = 1;
          * key_location = location;
          return;
        }
      }
      
      //Entry not found. Return NULL key.
      * key_found = 0;
      * key_location = UNDEFINED_KEY;
      
    }
    
    __inline__ __device__
    void
    hash_kernel(u32 id,
                u32* htable,
                u32* max_table,
                int table_size,
                u32* key_location,
                u32 max_age)
    {
      u32 location;
      u32 root_location;
      u32 evicted_key;
      u32 age = NULL_AGE;
      u32 key = pack_key_pos_and_age(id, KEY_TYPE_INIT_AGE);
      
      //Loop until max_age
      while(age < max_age){

        //evict key at current age-location
        //Update it if the to-be-inserted key is of a larger age
        location = hash(get_key_pos(key), age, hash_table_size);
        evicted_key = atomicMax(&htable[location], key);
        
        if(evicted_key < key){
          root_location = hash(get_key_pos(key), 0u, hash_table_size);
          atomicMax(&max_table[root_location], get_key_age(key));
  
          //If not an empty location, loop again to insert the evicted key.
          if(get_key_age(evicted_key) > 0u)
          {
            key = evicted_key;
            age = get_key_age(evicted_key);
          }
          //If empty location, finish the insertion.
          else
          {
            *key_location = get_key_pos(key);
            break;
          }
        }
        else
        {
          //Increase age and try again.
          age++;
          key = pack_key_pos_and_age(get_key_pos(key), age);
        }
      }
    }
    
    __global__
    void
    match(int m_size,
          int i_size,
          int hash_table_size,
          int* d_ck,
          int* d_inv,
          query::dim* d_dims,
          u32** hash_table_list,
          u32** count_table_list,
          float** aggregation_table_list,
          u32** age_table_list,
          u32 max_age)
    {
      query::dim* q = &d_dims[blockIdx.x];
      
      u32* hash_table = hash_table_list[blockIdx.x];
      u32* count_table = count_table_list[blockIdx.x];
      float* aggregation_table = aggregation_table_list[blockIdx.x];
      u32* age_table = age_table_list[blockIdx.x];

      int min, max;
      min = q->low;
      max = q->up;
      if (min > max)
        return;

      min < 1 ? min = 0 : min = d_ck[min - 1];
      max = d_ck[max];

      int loop = (max - min) / GaLG_device_THREADS_PER_BLOCK + 1;
      int part = blockIdx.x / m_size * i_size;

      int i;
      for (i = 0; i < loop; i++)
        {
          if (threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min < max)
            {
              u32 access_id = part + d_inv[threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min];
              u32 key_location;
              int key_found = 0;
              
              //Try to find the entry in hash tables
              access_kernel(access_id,
                            hash_table,
                            hash_table_size,
                            &key_location,
                            &key_found,
                            max_age);
              
              if(!key_found)
              {
                //Insert the key into count_table
                //access_id and its location are packed into a packed key
                //Create entry at the same location of aggregation_table
                hash_kernel(access_id,
                            hash_table,
                            age_table,
                            hash_table_size,
                            &key_location,
                            max_age);
              }
              atomicAdd(&count_table[key_location], 1);
              atomicAdd(&aggregation_table[key_location],q->weight);
            }
        }
    }
  }
}

void
GaLG::match(inv_table& table, vector<query>& queries,
    device_vector<int>& d_count,
    device_vector<float>& d_aggregation,
    device_vector<int>& d_hash,
            int& hash_table_size, int& ndims)
        throw (int)
{
  if (table.build_status() == inv_table::not_builded)
    throw inv_table::not_builded_exception;
  vector<query::dim> dims;
  int i;
  for (i = 0; i < queries.size(); i++)
    {
      if (queries[i].ref_table() != &table)
        throw inv_table::not_matched_exception;
      if (table.build_status() == inv_table::builded)
        queries[i].build();
      else if (table.build_status() == inv_table::builded_compressed)
        queries[i].build_compressed();
      queries[i].dump(dims);
    }
  ndims = dims.size();

  int total = table.i_size() * queries.size();

  device_vector<int> d_ck(*table.ck());
  int* d_ck_p = raw_pointer_cast(d_ck.data());

  device_vector<int> d_inv(*table.inv());
  int* d_inv_p = raw_pointer_cast(d_inv.data());

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());
  

  hash_table_size =(int)sqrt((double)total);
  
  u32** d_hash_table_list;
  cudaMalloc(&d_hash_table_list, sizeof(u32*)*dims.size());
  
  u32** d_count_table_list;
  cudaMalloc(&d_count_table_list, sizeof(u32*)*dims.size());
  
  float** d_aggregation_table_list;
  cudaMalloc(&d_aggregation_table_list, sizeof(float)*dims.size());
  
  u32** d_max_table_list;
  cudaMalloc(&d_max_table_list, sizeof(u32)*dims.size());
  
  for(i = 0; i < dims.size(); ++i){
    //hash table memory
    cudaMalloc(&d_hash_table_list[i], sizeof(u32)*hash_table_size);
    cudaMemset(&d_hash_table_list[i], UNDEFINED_KEY, sizeof(u32)*hash_table_size);
    //count table memory
    cudaMalloc(&d_count_table_list[i], sizeof(u32)*hash_table_size);
    cudaMemset(&d_count_table_list[i], 0u, sizeof(u32)*hash_table_size);
    //aggregation table memory
    cudaMalloc(&d_aggregation_table_list[i], sizeof(float)*hash_table_size);
    cudaMemset(&d_aggregation_table_list[i], 0f,sizeof(float)*hash_table_size);
    //max table memory
    cudaMalloc(&d_max_table_list[i], sizeof(u32)*hash_table_size);
    cudaMemset(&d_max_table_list[i], 0,sizeof(u32)*hash_table_size);
  }
  
  u32 max_age = 16u;
  
  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
  (table.m_size(),
   table.i_size(),
   hash_table_size,
   d_ck_p,
   d_inv_p,
   d_dims_p,
   d_hash_table_list,
   d_count_table_list,
   d_aggregation_table_list,
   d_max_table_list,
   max_age);
  
  d_count.resize(dims.size()*hash_table_size);
  d_aggregation.resize(dims.size()*hash_table_size);
  d_hash.resize(dims.size()*hash_table_size);
  
  for(i =0; i < dims.size(); ++i){
    
    thrust::copy(d_count_table_list[i],
                 d_count_table_list[i] + hash_table_size,
                 d_count.begin() + i * hash_table_size);
    
    thrust::copy(d_aggregation_table_list[i],
                 d_aggregation_table_list[i] + hash_table_size,
                 d_aggregation.begin() + i * hash_table_size);
    
    thrust::copy(d_hash_table_list[i],
                 d_hash_table_list[i] + hash_table_size,
                 d_hash.begin() + i * hash_table_size);
  }
  
}

void
GaLG::match(inv_table& table, query& queries, device_vector<int>& d_count,
    device_vector<float>& d_aggregation, device_vector<int>& d_hash, int& hash_table_size, int& ndims) throw (int)
{
  vector<query> _q;
  _q.push_back(queries);
  match(table, _q, d_count, d_aggregation, d_hash, hash_table_size, ndims);
}
