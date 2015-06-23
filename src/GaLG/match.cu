#include "match.h"
#include <cmath>
#include <sys/time.h>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
#endif

#define OFFSETS_TABLE_16 {0u,3949349u,8984219u,9805709u,7732727u,1046459u,9883879u,4889399u,2914183u,3503623u,1734349u,8860463u,1326319u,1613597u,8604269u,9647369u}

#define NULL_AGE 0

#define DEBUG
//#define DEBUG_VERBOSE

typedef u64 T_HASHTABLE;
typedef u32 T_KEY;
typedef u32 T_AGE;

u64 getTime()
{
 struct timeval tv;

 gettimeofday(&tv, NULL);

 u64 ret = tv.tv_usec;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000 * 1000);

 return ret;
}

float getInterval(u64 start, u64 stop)
{
	return ((float)(stop - start)) / 1000;
}

namespace GaLG
{
  namespace device
  {

     const u32 KEY_TYPE_BITS                 = 28u;
     const u32 KEY_TYPE_MASK                 = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 PACKED_KEY_TYPE_MASK          = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 KEY_TYPE_RANGE                = u32( u64((1ull) << KEY_TYPE_BITS) - 2u );
     const u32 UNDEFINED_KEY                 = u32( u64((1ull) << KEY_TYPE_BITS) - 1u );
     const u32 PACKED_UNDEFINED_KEY          = u32( u64((1ull) << KEY_TYPE_BITS) - 1ul);
    
     const u32 ATTACH_ID_TYPE_BITS           = 32u;
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
      return ((key) >> (KEY_TYPE_BITS)) & ATTACH_ID_TYPE_MASK;
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
                  int hash_table_size,
                  query::dim* q,
                  int * key_found,
                  u32 max_age)
    {
      u32 location;
      T_HASHTABLE out_key, new_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      
      location = hash(id, age, hash_table_size);

#ifdef DEBUG_VERBOSE
       printf(">>> [b%d t%d]Access: hash to %u. id: %u, age: %u.\n", blockIdx.x, threadIdx.x, location, id, age);
#endif
      while(1)
      {
          out_key = htable[location];

          if(get_key_pos(out_key) == id
          		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
          		&& get_key_age(out_key) < max_age){
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q->weight;
#ifdef DEBUG_VERBOSE
            printf("[b%dt%d] <Access1> new value: %f.\n", blockIdx.x, threadIdx.x,old_value_plus);
#endif
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =1;
            	return;
            }
          }
          else
          {
        	  break;
          }
      }

      //Key at root location is packed with its max age
      // in its hashing sequence.

      //Loop until max_age
      while(age < max_age){
    	age ++;
        location = hash(id, age, hash_table_size);
        out_key = htable[location];
        
#ifdef DEBUG_VERBOSE
        printf(">>> [b%d t%d]Access: hash to %u. id: %u, age: %u.\n", blockIdx.x, threadIdx.x, location, id, age);
#endif

        if(get_key_pos(out_key) == id
        		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
        		&& get_key_age(out_key) < max_age)
        {
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q->weight;
#ifdef DEBUG_VERBOSE
            printf("[b%dt%d] <Access2> new value: %f.\n", blockIdx.x, threadIdx.x,old_value_plus);
#endif
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =1;
#ifdef DEBUG_VERBOSE
            	attach_id = get_key_attach_id(htable[location]);
    			printf("[b%dt%d] <Access3> new value: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(&attach_id));
#endif
            	return;
            } else {
            	age --;
            	continue;
            }
        }


      }
      
      //Entry not found. Return NULL key.
      * key_found = 0;
      
    }
    
    
    __inline__ __device__
    void
    hash_kernel(u32 id,
                T_HASHTABLE* htable,
                int hash_table_size,
                query::dim* q,
                T_AGE max_age)
    {

      //u32 my_value = atomicAdd(value, 1);

#ifdef DEBUG_VERBOSE
      printf(">>> [b%d t%d]Insertion starts. weight is %f, Id is %d.\n", blockIdx.x, threadIdx.x, q->weight, id);
#endif

      //*value_index = my_value;
      
      u32 location;
      T_HASHTABLE evicted_key, peek_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      T_HASHTABLE key = pack_key_pos_and_attach_id_and_age(id,
                                                   *reinterpret_cast<u32*>(&(q->weight)),
                                                   KEY_TYPE_INIT_AGE);

      //Loop until max_age
      while(age < max_age){

        //evict key at current age-location
        //Update it if the to-be-inserted key is of a larger age
        location = hash(get_key_pos(key), age, hash_table_size);

        while(1)
        {
        	peek_key = htable[location];
        	if(get_key_pos(peek_key) == get_key_pos(key) && get_key_age(peek_key) != 0u)
        	{
        		u32 old_value_1 = get_key_attach_id(peek_key);
        		u32 old_value_2 = get_key_attach_id(key);
        		float old_value_plus = *reinterpret_cast<float*>(&old_value_2) + *reinterpret_cast<float*>(&old_value_1);
#ifdef DEBUG_VERBOSE
        		printf("[b%dt%d] <Hash1> new value: %f.\n", blockIdx.x, threadIdx.x, old_value_plus);
#endif
        		T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(peek_key),
        																 *reinterpret_cast<u32*>(&old_value_plus),
        																 get_key_age(peek_key));
        		if(atomicCAS(&htable[location], peek_key, new_key) == peek_key)
        		{
        			old_value_1 = get_key_attach_id(htable[location]);
#ifdef DEBUG_VERBOSE
        			printf("[b%dt%d] <Hash2> new value: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(&old_value_1));
#endif
        			return;
        		} else {
        			continue;
        		}
        	}
        	if(get_key_age(peek_key) < get_key_age(key))
        	{
        		evicted_key = atomicCAS(&htable[location], peek_key, key);
        		if(evicted_key != peek_key)
        			continue;
                if(get_key_age(evicted_key) > 0u)
                {
                  key = evicted_key;
                  age = get_key_age(evicted_key);
                  break;
                }
                else
                {
#ifdef DEBUG_VERBOSE
        			u32 old_value_1 = get_key_attach_id(htable[location]);
        			u64 keykey = htable[location];
        		    char s[65];
        		    s[64] = '\0';
        			for (int i = 63; i >= 0; i--)
        			    s[63 - i] = (keykey >> i) & 1 == 1? '1' : '0';

        			printf("[b%dt%d] <Hash3> new value: %f. Bit: %s.\n",
        					blockIdx.x,
        					threadIdx.x,
        					*reinterpret_cast<float*>(&old_value_1),
        					s);
#endif
                	return;
                }
        	}
        	else
        	{
                age++;
                key = pack_key_pos_and_attach_id_and_age(get_key_pos(key), get_key_attach_id(key), age);
                break;
        	}
        }

      }
      u32 attachid = get_key_attach_id(key);
      printf("[b%dt%d]Failed to update hash table. AGG: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(attachid));
    }
    
    __global__
    void
    match(int m_size,
          int i_size,
          int hash_table_size,
          int* d_ck,
          int* d_inv,
          query::dim* d_dims,
          T_HASHTABLE* hash_table_list,
          T_AGE max_age)
    {
      int query_index =blockIdx.x / m_size;
      query::dim* q = &d_dims[blockIdx.x];
      
      T_HASHTABLE* hash_table = &hash_table_list[query_index*hash_table_size];
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
                            q,
                            &key_found,
                            max_age);
              
              if(!key_found)
              {
                //Insert the key into hash table
                //access_id and its location are packed into a packed key
                hash_kernel(access_id,
                            hash_table,
                            hash_table_size,
                            q,
                            max_age);
              }
            }
        }
    }


	__global__
	void
	convert_to_data(T_HASHTABLE* table, u32 size)
	{
		u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= size) return;
		data_t *mytable = reinterpret_cast<data_t*>(&table[tid]);
		mytable->id = get_key_pos(table[tid]);
		u32 agg = get_key_attach_id(table[tid]);
		mytable->aggregation = *reinterpret_cast<float*>(&agg);
	}

  }
}
void
GaLG::match(inv_table& table,
            vector<query>& queries,
            device_vector<data_t>& d_data,
            int& hash_table_size)
throw (int)
{

	printf("match.cu version : %s\n", VERSION);
#ifdef DEBUG
	u64 match_stop, match_elapsed, match_start;
	cudaEvent_t kernel_start, kernel_stop;
	float kernel_elapsed;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	match_start = getTime();
	printf("[  0%] Starting matching...\n");
	printf("[ 10%] Fetching and packing data...\n");
#endif

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

#ifdef DEBUG
	printf("[ 20%] Declaring device memory...\n");
#endif

  device_vector<int> d_ck(*table.ck());
  int* d_ck_p = raw_pointer_cast(d_ck.data());

  device_vector<int> d_inv(*table.inv());
  int* d_inv_p = raw_pointer_cast(d_inv.data());

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());
  
  if(hash_table_size <= 0){
	  hash_table_size =table.i_size()/2 +1;
  }

  
#ifdef DEBUG
  printf("[ 30%] Allocating device memory to tables...\n");
#endif



  std::vector<T_HASHTABLE> h_hash_table(queries.size()*hash_table_size, 0ull);
  T_HASHTABLE* d_hash_table;
  data_t* d_data_table;
  d_data.clear();
  d_data.resize(queries.size()*hash_table_size);
  d_data_table = thrust::raw_pointer_cast(d_data.data());
  d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);
  cudaCheckErrors(cudaMemcpy(d_hash_table, &h_hash_table.front(), sizeof(T_HASHTABLE)*queries.size()*hash_table_size, cudaMemcpyHostToDevice));
  cudaCheckErrors(cudaDeviceSynchronize());
  h_hash_table.clear();

  u32 max_age = 16u;
  
#ifdef DEBUG
  printf("[ 33%] Copying memory to symbol...\n");
#endif

  u32 h_offsets[16] = OFFSETS_TABLE_16;
  
  cudaCheckErrors(cudaMemcpyToSymbol(GaLG::device::offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));


#ifdef DEBUG
  printf("[ 40%] Starting match kernels...\n");
  cudaEventRecord(kernel_start);
#endif

  device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
  (table.m_size(),
   table.i_size(),
   hash_table_size,
   d_ck_p,
   d_inv_p,
   d_dims_p,
   d_hash_table,
   max_age);
  
#ifdef DEBUG
  cudaEventRecord(kernel_stop);
  printf("[ 90%] Starting data converting......\n");
#endif

  cudaCheckErrors(cudaDeviceSynchronize());

  device::convert_to_data<<<hash_table_size*queries.size() / 1024 + 1, 1024>>>(d_hash_table,(u32)hash_table_size*queries.size());


#ifdef DEBUG
  printf("[100%] Matching is done!\n");

  match_stop = getTime();
  match_elapsed = match_stop - match_start;

  cudaEventSynchronize(kernel_stop);
  kernel_elapsed = 0.0f;
  cudaEventElapsedTime(&kernel_elapsed, kernel_start, kernel_stop);
  printf("[Info] Match function takes %f ms.\n", getInterval(match_start, match_stop));
  printf("[Info] Match kernel takes %f ms.\n", kernel_elapsed);

  printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
#endif
}

void
GaLG::match(inv_table& table,
            query& queries,
            device_vector<data_t>& d_data,
            int& hash_table_size)
throw (int)
{
  vector<query> _q;
  _q.push_back(queries);
  match(table, _q, d_data, hash_table_size);
}
