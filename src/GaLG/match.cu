#include "match.h"
#include <cmath>
#include <algorithm>

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
#endif

#define OFFSETS_TABLE_16 {0u,3949349u,8984219u,9805709u,7732727u,1046459u,9883879u,4889399u,2914183u,3503623u,1734349u,8860463u,1326319u,1613597u,8604269u,9647369u}

#define NULL_AGE 0

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef u64 T_HASHTABLE;
typedef u32 T_KEY;
typedef u32 T_AGE;


namespace GaLG
{
  namespace device
  {
     const u32 DEFAULT_GROUP_SIZE            = 192u;
    
     const u32 KEY_TYPE_BITS                 = 32u;
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
    
    __device__ __constant__ u32 offsets[16];
    
    __inline__ __host__ __device__
    T_KEY
    get_key_pos(T_HASHTABLE key)
    {
      return key & KEY_TYPE_MASK;
    }
    
    __inline__ __host__ __device__
    T_AGE
    get_key_age(T_HASHTABLE key)
    {
      return ((key) >> (ATTACH_ID_TYPE_BITS + KEY_TYPE_BITS));
    }

    __host__ __inline__ __device__
    u32
    get_key_attach_id(T_HASHTABLE key)
    {
      return ((k) >> (KEY_TYPE_BITS)) & ATTACH_ID_TYPE_MASK;
    }
    __host__ __inline__ __device__
    T_HASHTABLE
    pack_key_pos(T_KEY p)
    {
      return ((p) & KEY_TYPE_MASK);
    }
    __host__ __inline__ __device__
    T_HASHTABLE
    pack_key_pos_and_attach_id_and_age(T_KEY p, u32 i, T_AGE a)
    {
      return u64(((u64(a) << (ATTACH_ID_TYPE_BITS + KEY_TYPE_BITS))) + ((u64(i) & ATTACH_ID_TYPE_MASK) << (KEY_TYPE_BITS)) + u64(p & KEY_TYPE_MASK));
    }
    
    __inline__ __host__ __device__
    u32
    hash(T_KEY key, T_AGE age, int hash_table_size){
      return (offsets[age] + key) % hash_table_size;
    }
    
    
    __inline__ __device__
    void
    access_kernel(u32 id,
                  T_HASHTABLE* htable,
                  int table_size,
                  u32 * index,
                  int * key_found,
                  u32 max_age)
    {
      u32 location;
      T_HASHTABLE out_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      
      location = hash(id, age, hash_table_size);
      out_key = htable[location];
      
      if(get_key_pos(out_key) == id){
        * key_found = 1;
        * index = get_key_attach_id(out_key);
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
          * index = get_key_attach_id(out_key);
          return;
        }
      }
      
      //Entry not found. Return NULL key.
      * key_found = 0;
      * index = UNDEFINED_KEY;
      
    }
    
    
    __inline__ __device__
    void
    hash_kernel(u32 id,
                T_HASHTABLE* htable,
                T_AGE* max_table,
                int table_size,
                u32* value_index,
                T_AGE max_age,
                u32 * value)
    {

      u32 my_value = atomicAdd(value, 1);
      *value_index = my_value;
      
      u32 location;
      u32 root_location;
      T_HASHTABLE evicted_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      T_KEY key = pack_key_pos_and_attach_id_and_age(id,
                                                   my_value,
                                                   KEY_TYPE_INIT_AGE);
      
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
            *value_index = my_value;
            break;
          }
        }
        else
        {
          //Increase age and try again.
          age++;
          key = pack_key_pos_and_attach_id_and_age(get_key_pos(key), get_key_attach_id(key), age);
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
          T_HASHTABLE** hash_table_list,
          data** data_table_list,
          T_AGE** age_table_list,
          T_AGE max_age,
          u32 * value_idx)
    {
      int query_index =blockIdx.x / m_size;
      query::dim* q = &d_dims[blockIdx.x];
      
      T_HASHTABLE* hash_table = hash_table_list[query_index];
      T_AGE* age_table = age_table_list[query_index];
      data* data_table = data_table_list[query_index];
      
      u32 index, access_id;

      int min, max;
      min = q->low;
      max = q->up;
      if (min > max)
        return;

      min < 1 ? min = 0 : min = d_ck[min - 1];
      max = d_ck[max];

      int loop = (max - min) / GaLG_device_THREADS_PER_BLOCK + 1;


      int i;
      for (i = 0; i < loop; i++)
        {
          if (threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min < max)
            {
              access_id = d_inv[threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min];

              int key_found = 0;
              
              //Try to find the entry in hash tables
              access_kernel(access_id,
                            hash_table,
                            hash_table_size,
                            &index,
                            &key_found,
                            max_age);
              
              if(!key_found)
              {
                //Insert the key into hash table
                //access_id and its location are packed into a packed key
                hash_kernel(access_id,
                            hash_table,
                            age_table,
                            hash_table_size,
                            &index,
                            max_age,
                            value_idx);
              }
              data_table[index].id = access_id;
              atomicAdd(&(data_table[index].count), 1u);
              atomicAdd(&(data_table[index].aggregation),q->weight);
            }
        }
    }
  }
}

void
GaLG::match(inv_table& table,
            vector<query>& queries,
            device_vector<data>& d_data,
            int& hash_table_size)
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
  int total = table.i_size() * queries.size();

  device_vector<int> d_ck(*table.ck());
  int* d_ck_p = raw_pointer_cast(d_ck.data());

  device_vector<int> d_inv(*table.inv());
  int* d_inv_p = raw_pointer_cast(d_inv.data());

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());
  

  hash_table_size =(int)sqrt((double)total);
  
  T_HASHTABLE** d_hash_table_list;
  cudaMalloc(&d_hash_table_list, sizeof(T_HASHTABLE*)*queries.size());
  
  data** d_data_table_list;
  cudaMalloc(&d_data_table_list, sizeof(data*)*queries.size());
  
  T_AGE** d_max_table_list;
  cudaMalloc(&d_max_table_list, sizeof(T_AGE)*queries.size());
  
  for(i = 0; i < queries.size()(); ++i){
    //hash table memory
    cudaMalloc(&d_hash_table_list[i], sizeof(T_HASHTABLE)*hash_table_size);
    cudaMemset(&d_hash_table_list[i], UNDEFINED_KEY, sizeof(T_HASHTABLE)*hash_table_size);
    //data table memory
    cudaMalloc(&d_data_table_list[i], sizeof(data)*hash_table_size);
    cudaMemset(&d_data_table_list[i], 0u, sizeof(data)*hash_table_size);
    //max table memory
    cudaMalloc(&d_max_table_list[i], sizeof(T_AGE)*hash_table_size);
    cudaMemset(&d_max_table_list[i], 0,sizeof(T_AGE)*hash_table_size);
  }
  
  u32 max_age = 16u;
  
  u32 h_offsets[16] = OFFSETS_TABLE_16;
  
  if(!h_offsets_initialized)
  {
    cudaMemcpyToSymbol(offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice);
    h_offsets_initialized = 1;
  }
 
  
  u32 h_value_idx = 0;
  u32 * d_value_idx;
  cudaMalloc(&d_value_idx, sizeof(u32));
  cudaMemcpy(d_value_idx, &h_value_idx, sizeof(u32), cudaMemcpyHostToDevice);
  
  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK, sizeof(u32)>>>
  (table.m_size(),
   table.i_size(),
   hash_table_size,
   d_ck_p,
   d_inv_p,
   d_dims_p,
   d_hash_table_list,
   d_data_table_list,
   d_max_table_list,
   max_age,
   d_value_idx);
  
  cudaDeviceSynchronize();
  
  d_data.resize(dims.size()*hash_table_size);

  for(i =0; i < queries.size(); ++i){
    
    thrust::copy(d_data_table_list[i],
                 d_data_table_list[i] + hash_table_size,
                 d_data.begin() + i * hash_table_size);
    cudaFree(d_data_table_list[i]);
    cudaFree(d_hash_table_list[i]);
    cudaFree(d_max_table_list[i]);
  }
  
  cudaFree(d_data_table_list);
  cudaFree(d_hash_table_list);
  cudaFree(d_max_table_list);
  
}

void
GaLG::match(inv_table& table,
            query& queries,
            device_vector<data>& d_data,
            int& hash_table_size)
throw (int)
{
  vector<query> _q;
  _q.push_back(queries);
  match(table, _q, d_data, hash_table_size);
}
