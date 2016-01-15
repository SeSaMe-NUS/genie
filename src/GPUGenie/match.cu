#include "match.h"
#include <cmath>
#include <sys/time.h>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include "Logger.h"

#ifndef GPUGenie_device_THREADS_PER_BLOCK
#define GPUGenie_device_THREADS_PER_BLOCK 256
#endif

#define OFFSETS_TABLE_16 {0u,3949349u,8984219u,9805709u,7732727u,1046459u,9883879u,4889399u,2914183u,3503623u,1734349u,8860463u,1326319u,1613597u,8604269u,9647369u}

#define NULL_AGE 0
#define MAX_AGE 16u

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

double getInterval(u64 start, u64 stop)
{
	return ((double)(stop - start)) / 1000;
}

namespace GPUGenie
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
    
    /**
     *  @brief get the item id
     */
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
    get_key_attach_id(T_HASHTABLE key)//to get the count of one item
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
    
    __inline__ __device__ __host__
    void print_binary(char * b, u32 data)
    {
    	for (int i = 31; i >= 0; i--)
        	b[31-i] = ((data >> i) & 1) == 1 ? '1' : '0';
        b[32] = '\0';
    }

    __device__ __host__ __inline__
    u32
    get_count(u32 data, int offset, int bits)
    {
    	return (data >> offset) & ((1u << bits) - 1u);
    }

    __device__ __host__ __inline__
    u32
    pack_count(u32 data, int offset, int bits, u32 count)
    {
    	u32 r;
    	r = data & (~(((1u << bits) - 1u) << offset));
    	r |= (count << offset);
    	return r;
    }

    __inline__ __device__
    void
    access_kernel(u32 id,
                  T_HASHTABLE* htable,
                  int hash_table_size,
                  query::dim& q,
                  bool * key_found)
    {
      u32 location;
      T_HASHTABLE out_key, new_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      
      location = hash(id, age, hash_table_size);

      while(1)
      {
          out_key = htable[location];

          if(get_key_pos(out_key) == id
          		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
          		&& get_key_age(out_key) < MAX_AGE){
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =true;//
            	return;
            }
          }
          else
          {
        	  break;
          }
      }

      while(age < MAX_AGE){
    	age ++;
        location = hash(id, age, hash_table_size);
        out_key = htable[location];

        if(get_key_pos(out_key) == id
        		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
        		&& get_key_age(out_key) < MAX_AGE)
        {
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =true;
            	return;
            } else {
            	age --;
            	continue;
            }
        }
      }
      
      //Entry not found.
      * key_found = 0;
    }
    
    //for AT: for adaptiveThreshold
    __inline__ __device__
       void
       access_kernel_AT(u32 id,
                     T_HASHTABLE* htable,
                     int hash_table_size,
                     query::dim& q,
                     u32 count,
                     bool * key_found,
                     u32* my_threshold,
                     bool * pass_threshold // if the count smaller that my_threshold, do not insert
                     )
       {
         u32 location;
         T_HASHTABLE out_key, new_key;
         T_AGE age = KEY_TYPE_NULL_AGE;

         location = hash(id, age, hash_table_size);
         while(1)
         {
             out_key = htable[location];

             if(get_key_pos(out_key) == id
             		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
             		&& get_key_age(out_key) < MAX_AGE){
            	 u32 attach_id = get_key_attach_id(out_key);//for AT: for adaptiveThreshold
            	 //float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;//for AT: for adaptiveThreshold;   for improve: update here for weighted distance
            	 float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold
            	 //float value_plus = (value_1>count)? (value_1) : (count);//for AT:   for improve: update here for weighted distance
            	 float value_plus = count;//for AT: for adaptiveThreshold
            	 if(value_plus <value_1){//for AT: for adaptiveThreshold
            		 *pass_threshold = true;// still need to update the my_threshold and passCount
            		 *key_found =true;//already find the key, but do not update
            		 return;
            	 }
               new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            		   	   	   	   	   	   	   	   	   	*reinterpret_cast<u32*>(&value_plus),
               											 get_key_age(out_key));
               if(value_plus<*my_threshold){
               	*pass_threshold = false;// if my_threshold is updated, no need to update hash_table and threshold
               	*key_found =true;//already find the key, but do not update
                 return;//
               }
               if(atomicCAS(&htable[location], out_key, new_key) == out_key)
               {*pass_threshold = true;//high possible that pass the threshold, must update the threshold
               	*key_found =true;//
               	return;
               }
             }
             else
             {
           	  break;
             }
         }

         while(age < MAX_AGE){
           age ++;
           location = hash(id, age, hash_table_size);
           out_key = htable[location];

           if(get_key_pos(out_key) == id
           		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
           		&& get_key_age(out_key) < MAX_AGE)
           {
        	   u32 attach_id = get_key_attach_id(out_key);//for AT: for adaptiveThreshold
               //float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;//for AT: for adaptiveThreshold

        	   float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold  //for improve: update here for weighted distance
        	   //float value_plus = (value_1>count)? (value_1) : (count);//for AT:
        	   float value_plus = count;//for AT: for adaptiveThreshold
        	   if(value_plus <value_1){//for AT: for adaptiveThreshold
        		  *pass_threshold = true;// still need to update the my_threshold and passCount
        		  *key_found =true;//already find the key, but do not update
        	      return;
        	   }

               new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
               											 *reinterpret_cast<u32*>(&value_plus),//for impprove:update here for weighted distance
               											 get_key_age(out_key));
               if(value_plus<*my_threshold){
                  *pass_threshold = false;// if my_threshold is updated, no need to update hash_table and threshold
                  *key_found =true;//already find the key, but do not update
                  return;//
               }
               if(atomicCAS(&htable[location], out_key, new_key) == out_key)
               {
            	*pass_threshold = true;
               	*key_found =true;
               	return;
               } else {
               	age --;
               	continue;
               }
           }
         }

         //Entry not found.
         * key_found = false;
         //key not found, no need to update my_threshold
         *pass_threshold = false;
       }
    //for AT: for adaptiveThreshold

    __inline__ __device__
    void
    hash_kernel(u32 id,//
                T_HASHTABLE* htable,
                int hash_table_size,
                query::dim& q,
                u32 * my_noiih,
                bool * overflow)
    {
      u32 location;
      T_HASHTABLE evicted_key, peek_key;
      T_AGE age = KEY_TYPE_NULL_AGE;
      T_HASHTABLE key = pack_key_pos_and_attach_id_and_age(id,
                                                   *reinterpret_cast<u32*>(&(q.weight)),
                                                   KEY_TYPE_INIT_AGE);
      //Loop until MAX_AGE
      while(age < MAX_AGE){

        //evict key at current age-location
        //Update it if the to-be-inserted key is of a larger age
        location = hash(get_key_pos(key), age, hash_table_size);
        while(1)
        {
        	if(*my_noiih > hash_table_size)
        	{
        		*overflow = true;
        		return;
        	}

        	peek_key = htable[location];
        	//for parallel race region, item may be inserted by othe threads
        	if(get_key_pos(peek_key) == get_key_pos(key) && get_key_age(peek_key) != 0u)//
        	{
        		u32 old_value_1 = get_key_attach_id(peek_key);
        		u32 old_value_2 = get_key_attach_id(key);//
        		float old_value_plus = *reinterpret_cast<float*>(&old_value_2) + *reinterpret_cast<float*>(&old_value_1);
        		T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(peek_key),
        																 *reinterpret_cast<u32*>(&old_value_plus),
        																 get_key_age(peek_key));
        		if(atomicCAS(&htable[location], peek_key, new_key) == peek_key)
        		{
        			return;
        		} else {
        			continue;
        		}
        	}

        	if(get_key_age(peek_key) < get_key_age(key))//
        	{
        		evicted_key = atomicCAS(&htable[location], peek_key, key);
        		if(evicted_key != peek_key)
        			continue;
                if(get_key_age(evicted_key) > 0u)//
                {
                  key = evicted_key;
                  age = get_key_age(evicted_key);
                  break;
                }
                else//if get_key_age(evicted_key) == 0, this is empty insertion
                {

                	if(*my_noiih >= hash_table_size)
                	{
                		*overflow = true;
                		atomicAdd(my_noiih, 1u);//for improve: it can be improved: but it seems not delayed the performance too much
                		return;
                	} else{
                		atomicAdd(my_noiih, 1u);//for improve:
                	}


/*** DEBUG SECTION - DON'T REMOVE! ****/
/*** The debug section below can print out the key in binary. ****/
/*
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
*/

                	return;// finish insertion for empty location
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
      *overflow = true;
      return;
    }

    //for AT: for countHeap (with adaptiveThreshold)
    __inline__ __device__
       void
       hash_kernel_AT(u32 id,//
                   T_HASHTABLE* htable,
                   int hash_table_size,
                   query::dim& q,
                   u32 count,
                   u32* my_threshold,//for AT: for adaptiveThreshold, if the count is smaller than my_threshold, this item is also expired in the hashTable
                   u32 * my_noiih,
                   bool * overflow,
                   bool* pass_threshold
                   )
       {
         u32 location;
         T_HASHTABLE evicted_key, peek_key;
         T_AGE age = KEY_TYPE_NULL_AGE;
         float count_value = count;
         T_HASHTABLE key = pack_key_pos_and_attach_id_and_age(id,
                                                      //*reinterpret_cast<u32*>(&(q.weight)),//for AT: for adaptiveThreshold
        		 	 	 	 	 	 	 	 	 	 *reinterpret_cast<u32*>(&count_value),
                                                      KEY_TYPE_INIT_AGE);
         //Loop until MAX_AGE
         while(age < MAX_AGE){

           //evict key at current age-location
           //Update it if the to-be-inserted key is of a larger age
           u32 key_attach_id = get_key_attach_id(key);//for AT: for daptiveThreshold for ask: what is old_value_1, and what is old_value_2
           float key_value = *reinterpret_cast<float*>(&key_attach_id);
           if(key_value<*my_threshold){//no need to update
        	   if(get_key_pos(key)==id){
        		   *pass_threshold = false;//  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
        	   }else{
        		   *pass_threshold = true;//the id has been inserted into hashtable, this key_attach_id is from the evicted_key
        	   }
               return;
           }

           location = hash(get_key_pos(key), age, hash_table_size);
           while(1)
           {
           	if(*my_noiih > hash_table_size)
           	{
           		*overflow = true;
           		return;
           	}

           	peek_key = htable[location];
           	u32 peek_key_attach_id = get_key_attach_id(peek_key);//for AT: for adaptiveThreshold
           	float peek_key_value = *reinterpret_cast<float*>(&peek_key_attach_id);
           	if(get_key_pos(peek_key) == get_key_pos(key) && get_key_age(peek_key) != 0u)//even previously key_eligible ==0, the key may be inserted by other threads
           	{


           		//float old_value_plus = (old_value_1>old_value_2)? (*reinterpret_cast<float*>(&old_value_1)) : (*reinterpret_cast<float*>(&old_value_2));//for AT: for adaptiveThreshold


           		//float old_value_plus = (old_value_1>old_value_2)? (old_value_1) : (old_value_2);//for AT: for adaptiveThreshold
           		if(key_value<peek_key_value){//no need to update
           			*pass_threshold = true;// still need to update the my_threshold and passCount
           		     return;
           		}

           		T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(peek_key),
           																 *reinterpret_cast<u32*>(&key_value),//for improve: update here for weighted distance
           																 get_key_age(peek_key));

           		if(key_value<*my_threshold){//no need to update
           		 if(get_key_pos(key)==id){
           		      *pass_threshold = false;//  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
           		  }else{
           		      *pass_threshold = true;//the id has been inserted into hashtable, this key_attach_id is from the evicted_key
           		 }
           		    return;
           		}
           		if(atomicCAS(&htable[location], peek_key, new_key) == peek_key)
           		{

           			*pass_threshold = true;//after updat the hashtable, increase the pass_count and my_threshold
           			return;
           		} else {
           			continue;
           		}
           	}

           	if((get_key_age(peek_key) < get_key_age(key) //if this location with smaller age (inclusive empty location, i.e. age 0)
           			||(get_key_age(peek_key)!= KEY_TYPE_NULL_AGE&&peek_key_value<*my_threshold))//for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
           															//this item is also expired in the hashTable,
           			)
           	{
           		if(key_value<*my_threshold){//no need to update
           		    if(get_key_pos(key)==id){
           		       *pass_threshold = false;//  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
           		     }else{
           		       *pass_threshold = true;//the id has been inserted into hashtable, this key_attach_id is from the evicted_key
           		     }
           		       return;
           		  }

           		evicted_key = atomicCAS(&htable[location], peek_key, key);




           		if(evicted_key != peek_key)
           			continue;


                   if((get_key_age(evicted_key) > 0u)//if this not an empty location
                		   )
                   {
                	 if(peek_key_value<*my_threshold){// for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
													//this item is also expired in the hashTable,
                		 *pass_threshold = true;//after updating the hashtable, increase the pass_count and my_threshold
                		 return;//
                	 }


                     key = evicted_key;
                     age = get_key_age(evicted_key);

                     break;
                   }
                   else//if get_key_age(evicted_key) == 0, this is empty insertion, nothing need to do
                   {

                   	if(*my_noiih >= hash_table_size)
                   	{
                   		*overflow = true;
                   		atomicAdd(my_noiih, 1u);// this will not affect the performance very much
                   		return;
                   	} else{
                   		atomicAdd(my_noiih, 1u);// this will not affect the performance very much
                   	}
           			*pass_threshold = true;//after updating the hashtable, increase the pass_count and my_threshold

           			return;//finish insertion for empty location
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
         //u32 attachid = get_key_attach_id(key);
         //printf("[b%dt%d]Failed to update hash table. AGG: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(&attachid));
         *overflow = true;
         *pass_threshold = true;
         return;
       }
    //for AT: for adaptiveThreshold

    __device__ __inline__
    void
    bitmap_kernel(u32 access_id,
  		  	      u32 * bitmap,
  		  	      int bits,
  		  	      int threshold,
  		  	      bool * key_eligible)
    {
    	u32 value, count, new_value;
    	int offset;
    	while(1)
    	{
        	value = bitmap[access_id / (32 / bits)];
        	offset = (access_id % (32 / bits))*bits;
        	count = get_count(value, offset, bits);
        	if(count < threshold)
        	{
        		*key_eligible = false;
        		count ++;

        	}else
        	{
        		*key_eligible = true;
        		return;
        	}
        	new_value = pack_count(value, offset, bits, count);
        	if(atomicCAS(&bitmap[access_id / (32 / bits)], value, new_value) == value)
        		return;
    	}

    }

    //for AT: for adaptiveThreshold, this is function for bitmap
    __device__ __inline__
       u32
       bitmap_kernel_AT(u32 access_id,
     		  	      u32 * bitmap,
     		  	      int bits,
     		  	      int my_threshold,
     		  	      bool * key_eligible)
       {
       	u32 value, count, new_value;
       	int offset;
       	while(1)
       	{
           	value = bitmap[access_id / (32 / bits)];
           	offset = (access_id % (32 / bits))*bits;
           	count = get_count(value, offset, bits);
           	count =count+1; //always maintain the count in bitmap//for improve: change here for weighted distance
           	if(count < my_threshold)
           	{
           		*key_eligible = false;

           	}else
           	{
           		*key_eligible = true;

           	}
           	new_value = pack_count(value, offset, bits, count);
           	if(atomicCAS(&bitmap[access_id / (32 / bits)], value, new_value) == value)
           		return count;
       	}
       	return 0;//fail to access the count

       }
    //for AT: for adaptiveThreshold, this is function for bitmap

    __global__
    void
    match(int m_size,
          int i_size,
          int hash_table_size,
          int* d_ck,
          int* d_inv,
          int* d_inv_index,
          int* d_inv_pos,
          query::dim* d_dims,
          T_HASHTABLE* hash_table_list,
          u32 * bitmap_list,
          int bitmap_bits,
          int threshold,
          u32 * noiih,
          bool * overflow)
    {
      if(m_size == 0 || i_size == 0) return;
      query::dim& q = d_dims[blockIdx.x];
      int query_index = q.query;
      u32* my_noiih = &noiih[query_index];
      
      T_HASHTABLE* hash_table = &hash_table_list[query_index*hash_table_size];
      u32 * bitmap;
      if(bitmap_bits) bitmap = &bitmap_list[query_index * (i_size / (32 /bitmap_bits) + 1)];
      u32 access_id;

      int min, max, min_offset, max_offset;
      min = q.low;
      min_offset = q.low_offset;
      max = q.up;
      max_offset = q.up_offset;
      if (min > max)
        return;

      min = d_inv_pos[d_inv_index[min]+min_offset];
      max = d_inv_pos[d_inv_index[max]+max_offset+1];

      bool key_eligible;

      for (int i = 0; i < (max - min) / GPUGenie_device_THREADS_PER_BLOCK + 1; i++)
        {
    	  int tmp_id = threadIdx.x + i * GPUGenie_device_THREADS_PER_BLOCK + min;
          if (tmp_id < max)
            {
              access_id = d_inv[tmp_id];
              if(bitmap_bits){
            	  key_eligible = false;
                  bitmap_kernel(access_id,
                		  	    bitmap,
                		  	    bitmap_bits,
                		  	    threshold,
                		  	    &key_eligible);

                  if( !key_eligible ) continue;
              }

              key_eligible = false;
              //Try to find the entry in hash tables
              access_kernel(access_id,//
                            hash_table,
                            hash_table_size,
                            q,
                            &key_eligible);
              
              if(!key_eligible)
              {
                //Insert the key into hash table
                //access_id and its location are packed into a packed key
                hash_kernel(access_id,
                            hash_table,
                            hash_table_size,
                            q,
                            my_noiih,
                            overflow);
                if(*overflow)
                {
                	return;
                }
              }
            }
        }
    }

	__device__ inline void updateThreshold(u32* my_passCount,u32* my_threshold,u32  my_topk, u32 count) {
		if(count< *my_threshold){
			 return;//threshold has been increased, no need to update
		}
		atomicAdd(&my_passCount[count], 1);                //successfully update

		u32 this_threshold = (*my_threshold);

		while (true) {
			this_threshold = *my_threshold;
			if (my_passCount[this_threshold] >= my_topk) {
				this_threshold = atomicCAS(my_threshold, this_threshold,
						this_threshold + 1);
			} else {
				break;
			}
		}
	}
    //for AT: for adaptiveThreshold match function for adaptiveThreshold
    __global__
     void
     match_AT(int m_size,
               int i_size,
               int hash_table_size,
               int* d_ck,
               int* d_inv,
               int* d_inv_index,
               int* d_inv_pos,
               query::dim* d_dims,
               T_HASHTABLE* hash_table_list,
               u32 * bitmap_list,
               int bitmap_bits,
               u32* d_topks,
               u32* d_threshold,//initialized as 1, and increase gradually
               u32* d_passCount,//initialized as 0, count the number of items passing one d_threshold
               u32 num_of_max_count,
               u32 * noiih,
               bool * overflow)
     {
       if(m_size == 0 || i_size == 0) return;
       query::dim& q = d_dims[blockIdx.x];
       int query_index = q.query;
       u32* my_noiih = &noiih[query_index];
       u32* my_threshold = &d_threshold[query_index];
       u32* my_passCount = &d_passCount[query_index*num_of_max_count];//
       u32  my_topk = d_topks[query_index];//for AT

       T_HASHTABLE* hash_table = &hash_table_list[query_index*hash_table_size];
       u32 * bitmap;
       if(bitmap_bits) bitmap = &bitmap_list[query_index * (i_size / (32 /bitmap_bits) + 1)];
       u32 access_id;

       int min, max, min_offset, max_offset;
       min = q.low;
       min_offset = q.low_offset;
       max = q.up;
       max_offset = q.up_offset;
       if (min > max)
         return;

       min = d_inv_pos[d_inv_index[min]+min_offset];
       max = d_inv_pos[d_inv_index[max]+max_offset+1];

       bool key_eligible;//
       bool pass_threshold;//to determine whether pass the check of my_theshold

       for (int i = 0; i < (max - min) / GPUGenie_device_THREADS_PER_BLOCK + 1; i++)
         {
     	  int tmp_id = threadIdx.x + i * GPUGenie_device_THREADS_PER_BLOCK + min;
           if (tmp_id < max)
             {
        	   u32 count = 0;//for AT
               access_id = d_inv[tmp_id];
               u32 thread_threshold = *my_threshold;
               if(bitmap_bits){

             	  key_eligible = false;
             	  //all count are store in the bitmap, and access the count
                  count = bitmap_kernel_AT(access_id,
                 		  	    bitmap,
                 		  	    bitmap_bits,
                 		  	    thread_threshold,
                 		  	    &key_eligible);

                   if( !key_eligible) continue;//i.e. count< thread_threshold
               }


						key_eligible = false;
						if(count< *my_threshold){
							continue;//threshold has been increased, no need to insert
						}

					   //Try to find the entry in hash tables
					   access_kernel_AT(access_id,//
									 hash_table,
									 hash_table_size,
									 q,
									 count,
									 &key_eligible,
									 my_threshold,
									 &pass_threshold
									 );

					   if(key_eligible){
						   if(pass_threshold){
							   updateThreshold(my_passCount,my_threshold, my_topk,count);
						   }

						   continue;
					   }

					   if(!key_eligible)
					   {
						 //Insert the key into hash table
						 //access_id and its location are packed into a packed key

						 if(count< *my_threshold){
							 continue;//threshold has been increased, no need to insert
						 }


						 hash_kernel_AT(access_id,
									 hash_table,
									 hash_table_size,
									 q,
									 count,
									 my_threshold,
									 my_noiih,
									 overflow,
									 &pass_threshold);
						 if(*overflow)
						 {

							return;
						 }
						 if(pass_threshold){
							 updateThreshold(my_passCount,my_threshold, my_topk,count);
						 }
					   }


             }
         }
     }
    //end for AT

	__global__
	void
	convert_to_data(T_HASHTABLE* table, u32 size)
	{
		u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= size) return;
		data_t *mytable = reinterpret_cast<data_t*>(&table[tid]);
		u32 agg = get_key_attach_id(table[tid]);
		u32 myid = get_key_pos(table[tid]);
		mytable->id = myid;
		mytable->aggregation = *reinterpret_cast<float*>(&agg);
	}

  }
}

void
GPUGenie::build_queries(vector<query>& queries, inv_table& table, vector<query::dim>& dims, int max_load)
{
	for (int i = 0; i < queries.size(); ++i)
	{
	  if (queries[i].ref_table() != &table)
		throw inv_table::not_matched_exception;
	  if (table.build_status() == inv_table::builded)
		  if(queries[i].use_load_balance)
		  {
			  queries[i].build_and_apply_load_balance(max_load);

		  } else {
			  queries[i].build();
		  }

	  else if (table.build_status() == inv_table::builded_compressed)
		queries[i].build_compressed();

	  queries[i].dump(dims);
	}
}
void
GPUGenie::match(inv_table& table,
            vector<query>& queries,
            device_vector<data_t>& d_data,
            int hash_table_size,
            int max_load,
            int bitmap_bits,
            device_vector<u32>& d_noiih)
{
	device_vector<u32> d_bitmap;
	match(table, queries,d_data,d_bitmap,hash_table_size,max_load,bitmap_bits, d_noiih);
}

void
GPUGenie::match(inv_table& table,
            vector<query>& queries,
            device_vector<data_t>& d_data,
            device_vector<u32>& d_bitmap,
            int hash_table_size,
            int max_load,
            int bitmap_bits,//or for AT: for adaptiveThreshold, if bitmap_bits<0, use adaptive threshold, the absolute value of bitmap_bits is count value stored in the bitmap
            device_vector<u32>& d_noiih)
{
	u64 match_stop, match_elapsed, match_start;
	cudaEvent_t kernel_start, kernel_stop;
	float kernel_elapsed;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	match_start = getTime();
	Logger::log(Logger::INFO,"[  0%] Starting matching...");

	if (table.build_status() == inv_table::not_builded)
		throw inv_table::not_builded_exception;

	u32 loop_count = 0u;
	d_noiih.resize(queries.size(),0);
	//thrust::fill(d_noiih.begin(), d_noiih.end(), 0u);//for improve
	u32 * d_noiih_p = thrust::raw_pointer_cast(d_noiih.data());

	vector<query::dim> dims;
	vector<query::dim> hot_dims;
	vector<query> hot_dims_queries;

	Logger::log(Logger::DEBUG,"hash table size: %d.", hash_table_size);
  u64 match_query_start,match_query_end;
  match_query_start=getTime();

  build_queries(queries, table, dims, max_load);

  match_query_end=getTime();
  Logger::log(Logger::VERBOSE,">>>>[time profiling]: match: build_queries function takes %f ms. ", getInterval(match_query_start, match_query_end));
  Logger::log(Logger::DEBUG," dims size: %d. hot_dims size: %d.", dims.size(), hot_dims.size());

  //for AT: for adaptiveThreshold, enable adaptiveThreshold
  bool useAdaptiveThreshold = false;//for AT
  if(bitmap_bits<0){
	 bitmap_bits = -bitmap_bits;
	 useAdaptiveThreshold = true;
	 //for hash_table_size, still let it determine by users currently
  }

  Logger::log(Logger::DEBUG,"[info] useAdaptiveThreshold: %d, bitmap_bits:%d.", useAdaptiveThreshold, bitmap_bits);

  //end for AT

  int total = table.i_size() * queries.size();
  int threshold = bitmap_bits - 1, bitmap_size = 0, bitmap_bytes = 0;
  if(bitmap_bits > 1){
	  float logresult = log2((float) bitmap_bits);
	  bitmap_bits = (int) logresult;
	  if(logresult - bitmap_bits > 0)
	  {
		 bitmap_bits += 1;
	  }
	  logresult = log2((float)bitmap_bits);
	  bitmap_bits = (int) logresult;
	  if(logresult - bitmap_bits > 0)
	  {
		 bitmap_bits += 1;
	  }
	  bitmap_bits = pow(2, bitmap_bits);
	  bitmap_size = (table.i_size() / (32/bitmap_bits) + 1)* queries.size();
	  bitmap_bytes = bitmap_size * sizeof(u32);
  }
  else
  {
	  bitmap_bits = threshold = 0;
  }

  Logger::log(Logger::DEBUG,"Bitmap bits: %d, threshold:%d.", bitmap_bits, threshold);
  Logger::log(Logger::INFO,"[ 20%] Declaring device memory...");

  d_bitmap.resize(bitmap_size);

  device_vector<query::dim> d_dims(dims);
  query::dim* d_dims_p = raw_pointer_cast(d_dims.data());

  if(!table.is_stored_in_gpu)
       table.cpy_data_to_gpu();

	if(bitmap_size)
	{
		thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
	}
	u32 * d_bitmap_p = raw_pointer_cast(d_bitmap.data());

	Logger::log(Logger::INFO,"[ 30%] Allocating device memory to tables...");

	data_t nulldata;
	nulldata.id = 0u;
	nulldata.aggregation = 0.0f;
	T_HASHTABLE* d_hash_table;
	data_t* d_data_table;
	d_data.clear();

	d_data.resize(queries.size()*hash_table_size,nulldata);
	//thrust::fill(d_data.begin(), d_data.end(), nulldata);//for imp
	d_data_table = thrust::raw_pointer_cast(d_data.data());
	d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);

	Logger::log(Logger::INFO,"[ 33%] Copying memory to symbol...");

  u32 h_offsets[16] = OFFSETS_TABLE_16;
  cudaCheckErrors(cudaMemcpyToSymbol(GPUGenie::device::offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));

  Logger::log(Logger::INFO,"[ 40%] Starting match kernels...");
  cudaEventRecord(kernel_start);

  	bool h_overflow[1]= {false};
    bool * d_overflow;
    cudaCheckErrors(cudaMalloc((void**) &d_overflow, sizeof(bool)));


	do{
		h_overflow[0] = false;
		cudaCheckErrors(cudaMemcpy(d_overflow, h_overflow, sizeof(bool), cudaMemcpyHostToDevice));
		//cudaCheckErrors(cudaDeviceSynchronize());
		u32 num_of_max_count=0,max_topk=0;
		if(!useAdaptiveThreshold) //for AT: for adaptiveThreshold, branch here
		{
		device::match<<<dims.size(), GPUGenie_device_THREADS_PER_BLOCK>>>
					(table.m_size(),
					table.i_size(),
					hash_table_size,
					table.d_ck_p,
					table.d_inv_p,
					table.d_inv_index_p,
					table.d_inv_pos_p,
					d_dims_p,
					d_hash_table,
					d_bitmap_p,
					bitmap_bits,
					threshold,
					d_noiih_p,
					d_overflow);
        if(!table.is_stored_in_gpu)
            table.clear_gpu_mem();
	}else{//for AT: for adaptiveThreshold, use different match method for adaptiveThreshold



		device_vector<u32> d_threshold;
		d_threshold.resize(queries.size(),1);
		//thrust::fill(d_threshold.begin(), d_threshold.end(), 1);
		u32 * d_threshold_p = thrust::raw_pointer_cast(d_threshold.data());

		device_vector<u32> d_passCount;
		num_of_max_count = dims.size();
		d_passCount.resize(queries.size()*num_of_max_count,0u);//
		//thrust::fill(d_passCount.begin(), d_passCount.end(), 0u);
		u32 * d_passCount_p = thrust::raw_pointer_cast(d_passCount.data());

		//host_vector<u32> h_tops(queries.size());
		//max_topk = 100;
		//for (u32 i = 0; i < queries.size(); i++)
		// {
		//	  h_tops[i] = queries[i].topk();
		//	  if(h_tops[i]>max_topk){
		//		  max_topk = h_tops[i];
		//	  }
		//}
		// device_vector<u32> d_topks(h_tops);
		// u32 * d_topks_p = thrust::raw_pointer_cast(d_topks.data());
		max_topk = 100;
		device_vector<u32> d_topks;
		d_topks.resize(queries.size(),max_topk);
		u32 * d_topks_p = thrust::raw_pointer_cast(d_topks.data());


		device::match_AT<<<dims.size(), GPUGenie_device_THREADS_PER_BLOCK>>>
						(table.m_size(),
						table.i_size(),
						hash_table_size,
						table.d_ck_p,
						table.d_inv_p,
						table.d_inv_index_p,
						table.d_inv_pos_p,
						d_dims_p,
						d_hash_table,
						d_bitmap_p,
						bitmap_bits,
						d_topks_p,
						d_threshold_p,//initialized as 1, and increase gradually
						d_passCount_p,//initialized as 0, count the number of items passing one d_threshold
						num_of_max_count,//number of maximum count per query
					   d_noiih_p,
					   d_overflow);
        if(!table.is_stored_in_gpu)
            table.clear_gpu_mem();
	}//end for AT: for adaptiveThreshold, use different match method for adaptiveThreshold

		//cudaCheckErrors(cudaDeviceSynchronize());
		cudaCheckErrors(cudaMemcpy(h_overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));

		if(h_overflow[0])
		{	loop_count ++;
			if(!useAdaptiveThreshold){
				hash_table_size += 0.1f * table.i_size();
			}else{
				hash_table_size +=  num_of_max_count*max_topk;
			}
			if(hash_table_size > table.i_size())
			{
				hash_table_size = table.i_size();
			}
			thrust::fill(d_noiih.begin(), d_noiih.end(), 0u);
			if(bitmap_size)
			{
				thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
			}
			d_data.resize(queries.size()*hash_table_size);
			thrust::fill(d_data.begin(), d_data.end(), nulldata);
			d_data_table = thrust::raw_pointer_cast(d_data.data());
			d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);
		}

		if (loop_count>0){
			Logger::log(Logger::INFO,"%d time trying to launch match kernel: %s!", loop_count, h_overflow[0]?"failed":"succeeded");
		}

	} while(h_overflow[0]);

  cudaEventRecord(kernel_stop);
  Logger::log(Logger::INFO,"[ 90%] Starting data converting......");

  //cudaCheckErrors(cudaDeviceSynchronize());
  device::convert_to_data<<<hash_table_size*queries.size() / 1024 + 1, 1024>>>(d_hash_table,(u32)hash_table_size*queries.size());

  Logger::log(Logger::INFO,"[100%] Matching is done!");

  match_stop = getTime();
  match_elapsed = match_stop - match_start;

  cudaEventSynchronize(kernel_stop);
  kernel_elapsed = 0.0f;
  cudaEventElapsedTime(&kernel_elapsed, kernel_start, kernel_stop);
  Logger::log(Logger::VERBOSE,">>>>[time profiling]: Match kernel takes %f ms. (GPU running) ", kernel_elapsed);
  Logger::log(Logger::VERBOSE,">>>>[time profiling]: Match function takes %f ms.  (including Match kernel, GPU+CPU part)", getInterval(match_start, match_stop));
  Logger::log(Logger::VERBOSE,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
}
