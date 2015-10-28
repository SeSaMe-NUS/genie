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

#ifndef GaLG_device_THREADS_PER_BLOCK
#define GaLG_device_THREADS_PER_BLOCK 256
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
    get_key_pos(T_HASHTABLE key)//for ask: what does this function mean?//the item id
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

#ifdef DEBUG_VERBOSE
       printf(">>> [b%d t%d]Access: hash to %u. id: %u, age: %u.\n", blockIdx.x, threadIdx.x, location, id, age);
#endif
      while(1)
      {
          out_key = htable[location];

          if(get_key_pos(out_key) == id
          		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
          		&& get_key_age(out_key) < MAX_AGE){
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;
#ifdef DEBUG_VERBOSE
            printf("[b%dt%d] <Access1> new value: %f.\n", blockIdx.x, threadIdx.x,old_value_plus);
#endif
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =true;//for ask: why not combine access and insert?
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

#ifdef DEBUG_VERBOSE
        printf(">>> [b%d t%d]Access: hash to %u. id: %u, age: %u.\n", blockIdx.x, threadIdx.x, location, id, age);
#endif

        if(get_key_pos(out_key) == id
        		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
        		&& get_key_age(out_key) < MAX_AGE)
        {
        	u32 attach_id = get_key_attach_id(out_key);
            float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;
#ifdef DEBUG_VERBOSE
            printf("[b%dt%d] <Access2> new value: %f.\n", blockIdx.x, threadIdx.x,old_value_plus);
#endif
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            											 *reinterpret_cast<u32*>(&old_value_plus),
            											 get_key_age(out_key));
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
            	*key_found =true;
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
                     bool * key_found)
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
             		&& get_key_age(out_key) < MAX_AGE){
            	 u32 attach_id = get_key_attach_id(out_key);//for AT: for adaptiveThreshold
            	 //float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;//for AT: for adaptiveThreshold;   for improve: update here for weighted distance
            	 float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold
            	 float value_plus = (value_1>count)? (value_1) : (count);//for AT:   for improve: update here for weighted distance


   #ifdef DEBUG_VERBOSE
               printf("[b%dt%d] <Access1> new value: %f.\n", blockIdx.x, threadIdx.x,value_plus);
   #endif
               new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
            		   	   	   	   	   	   	   	   	   	*reinterpret_cast<u32*>(&value_plus),
               											 get_key_age(out_key));
               if(atomicCAS(&htable[location], out_key, new_key) == out_key)
               {
               	*key_found =true;//for ask: why not combine access and insert?
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

   #ifdef DEBUG_VERBOSE
           printf(">>> [b%d t%d]Access: hash to %u. id: %u, age: %u.\n", blockIdx.x, threadIdx.x, location, id, age);
   #endif

           if(get_key_pos(out_key) == id
           		&& get_key_age(out_key) != KEY_TYPE_NULL_AGE
           		&& get_key_age(out_key) < MAX_AGE)
           {
        	   u32 attach_id = get_key_attach_id(out_key);//for AT: for adaptiveThreshold
               //float old_value_plus = *reinterpret_cast<float*>(&attach_id) + q.weight;//for AT: for adaptiveThreshold

        	   float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold  //for improve: update here for weighted distance
        	   float value_plus = (value_1>count)? (value_1) : (count);//for AT:

   #ifdef DEBUG_VERBOSE
               printf("[b%dt%d] <Access2> new value: %f.\n", blockIdx.x, threadIdx.x,old_value_plus);
   #endif
               new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
               											 *reinterpret_cast<u32*>(&value_plus),//for impprove:update here for weighted distance
               											 get_key_age(out_key));
               if(atomicCAS(&htable[location], out_key, new_key) == out_key)
               {
               	*key_found =true;
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

         //Entry not found.
         * key_found = 0;
       }
    //for AT: for adaptiveThreshold


    __inline__ __device__
    void
    hash_kernel(u32 id,//for ask: understand this functoin is important
                T_HASHTABLE* htable,
                int hash_table_size,
                query::dim& q,
                u32 * my_noiih,
                bool * overflow)
    {
#ifdef DEBUG_VERBOSE
      printf(">>> [b%d t%d]Insertion starts. weight is %f, Id is %d.\n", blockIdx.x, threadIdx.x, q.weight, id);
#endif
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
        	if(get_key_pos(peek_key) == get_key_pos(key) && get_key_age(peek_key) != 0u)//for ask: where insert here? It seems it is impossible to satisfy this condition if key_eligible ==0
        	{
        		u32 old_value_1 = get_key_attach_id(peek_key);
        		u32 old_value_2 = get_key_attach_id(key);//for ask: what is old_value_1, and what is old_value_2
        		float old_value_plus = *reinterpret_cast<float*>(&old_value_2) + *reinterpret_cast<float*>(&old_value_1);
#ifdef DEBUG_VERBOSE
        		printf("[b%dt%d] <Hash1> new value: %f.\n", blockIdx.x, threadIdx.x, old_value_plus);
#endif
        		T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(peek_key),
        																 *reinterpret_cast<u32*>(&old_value_plus),
        																 get_key_age(peek_key));
        		if(atomicCAS(&htable[location], peek_key, new_key) == peek_key)
        		{
#ifdef DEBUG_VERBOSE
        			old_value_1 = get_key_attach_id(htable[location]);
        			printf("[b%dt%d] <Hash2> new value: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(&old_value_1));
#endif
        			return;
        		} else {
        			continue;
        		}
        	}

        	if(get_key_age(peek_key) < get_key_age(key))//for ask: if this location with smaller age (inclusive empty location, i.e. age 0)
        	{
        		evicted_key = atomicCAS(&htable[location], peek_key, key);
        		if(evicted_key != peek_key)
        			continue;
                if(get_key_age(evicted_key) > 0u)//for ask: if this not an empty location
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
                		atomicAdd(my_noiih, 1u);//for ask: this will affect the performance very much? //for improve:
                		return;
                	} else{
                		atomicAdd(my_noiih, 1u);//for improve:
                	}

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
                	return;//for ask: finish insertion for empty location
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
      return;
    }

    //for AT: for adaptiveThreshold
    __inline__ __device__
       void
       hash_kernel_AT(u32 id,//for ask: understand this functoin is important
                   T_HASHTABLE* htable,
                   int hash_table_size,
                   query::dim& q,
                   u32 count,
                   u32 my_threshold,//for AT: for adaptiveThreshold, if the count is smaller than my_threshold, this item is also expired in the hashTable
                   u32 * my_noiih,
                   bool * overflow)
       {
   #ifdef DEBUG_VERBOSE
         printf(">>> [b%d t%d]Insertion starts. weight is %f, Id is %d.\n", blockIdx.x, threadIdx.x, q.weight, id);
   #endif
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
           location = hash(get_key_pos(key), age, hash_table_size);
           while(1)
           {
           	if(*my_noiih > hash_table_size)
           	{
           		*overflow = true;
           		return;
           	}

           	peek_key = htable[location];
           	if(get_key_pos(peek_key) == get_key_pos(key) && get_key_age(peek_key) != 0u)//for ask: where insert here? It seems it is impossible to satisfy this condition if key_eligible ==0
           	{
           		u32 old_attach_id_1 = get_key_attach_id(peek_key);//for AT: for adaptiveThreshold
           		u32 old_attach_id_2 = get_key_attach_id(key);//for AT: for daptiveThreshold for ask: what is old_value_1, and what is old_value_2
           		//float old_value_plus = (old_value_1>old_value_2)? (*reinterpret_cast<float*>(&old_value_1)) : (*reinterpret_cast<float*>(&old_value_2));//for AT: for adaptiveThreshold
           		float old_value_1 = *reinterpret_cast<float*>(&old_attach_id_1);
           		float old_value_2 = *reinterpret_cast<float*>(&old_attach_id_2);
           		float old_value_plus = (old_value_1>old_value_2)? (old_value_1) : (old_value_2);//for AT: for adaptiveThreshold
   #ifdef DEBUG_VERBOSE
           		printf("[b%dt%d] <Hash1> new value: %f.\n", blockIdx.x, threadIdx.x, old_value_plus);
   #endif
           		T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(peek_key),
           																 *reinterpret_cast<u32*>(&old_value_plus),//for improve: update here for weighted distance
           																 get_key_age(peek_key));
           		if(atomicCAS(&htable[location], peek_key, new_key) == peek_key)
           		{
   #ifdef DEBUG_VERBOSE
           			old_value_1 = get_key_attach_id(htable[location]);
           			printf("[b%dt%d] <Hash2> new value: %f.\n", blockIdx.x, threadIdx.x, *reinterpret_cast<float*>(&old_value_1));
   #endif
           			//for debug
           			//old_attach_id_1 = get_key_attach_id(htable[location]);
           			//old_value_1 = *reinterpret_cast<float*>(&old_value_1);
           			//u32 new_attach_id = get_key_attach_id(new_key);
           			//float new_value = *reinterpret_cast<float*>(&new_attach_id);
           			//printf("[b%dt%d] <Hash2> new value: %.1f. with count=%d old_value_1=%.1f old_value_plus=%.1f new_value=%.1f\n", blockIdx.x, threadIdx.x, new_value,count,old_value_1, old_value_plus,new_value);
           			//end for debug
           			return;
           		} else {
           			continue;
           		}
           	}

           	if((get_key_age(peek_key) < get_key_age(key)) //if this location with smaller age (inclusive empty location, i.e. age 0)
           			||(get_key_attach_id(peek_key)<my_threshold)//for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
           															//this item is also expired in the hashTable, here !!!, for improve later
           			)
           	{
           		evicted_key = atomicCAS(&htable[location], peek_key, key);
           		if(evicted_key != peek_key)
           			continue;
                   if((get_key_age(evicted_key) > 0u)//for ask: if this not an empty location
                		   &&(get_key_attach_id(peek_key)>=my_threshold)//for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
																			//this item is also expired in the hashTable, here !!!, for improve later
                		   )
                   {
                     key = evicted_key;
                     age = get_key_age(evicted_key);
                     break;
                   }
                   else//if get_key_age(evicted_key) == 0, this is empty insertion, nothing need to do
                   {

                   	if(*my_noiih >= hash_table_size)
                   	{
                   		*overflow = true;
                   		atomicAdd(my_noiih, 1u);//for ask: this will affect the performance very much? //for improve:
                   		return;
                   	} else{
                   		atomicAdd(my_noiih, 1u);//for improve:
                   	}

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
                   	return;//for ask: finish insertion for empty location
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
         return;
       }
    //for AT: for adaptiveThreshold

    __device__ __inline__
    void
    bitmap_kernel(u32 access_id,
  		  	      u32 * bitmap,
  		  	      int bits,
  		  	      int threshold,
  		  	      bool * key_eligible,
  		  	      int num_of_hot_dims,
  		  	      int hot_dim_threshold)
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
        		if(num_of_hot_dims == 0 || count + num_of_hot_dims >= hot_dim_threshold)
        		{
        			count ++;
        		} else
        		{
        			return;
        		}
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
     		  	      bool * key_eligible,
     		  	      int num_of_hot_dims,
     		  	      int hot_dim_threshold)
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
          query::dim* d_dims,
          T_HASHTABLE* hash_table_list,
          u32 * bitmap_list,
          int bitmap_bits,
          int threshold,
          int num_of_hot_dims,
          int hot_dim_threshold,
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

      int min, max;
      min = q.low;
      max = q.up;
      if (min > max)
        return;

      min < 1 ? min = 0 : min = d_ck[min - 1];
      max = d_ck[max];

      bool key_eligible;//for ask: what does it mean for key_eligible

      for (int i = 0; i < (max - min) / GaLG_device_THREADS_PER_BLOCK + 1; i++)
        {
    	  int tmp_id = threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min;
          if (tmp_id < max)
            {
              access_id = d_inv[tmp_id];
              if(bitmap_bits){
            	  key_eligible = false;
                  bitmap_kernel(access_id,
                		  	    bitmap,
                		  	    bitmap_bits,
                		  	    threshold,
                		  	    &key_eligible,
                		  	    num_of_hot_dims,
                		  	    hot_dim_threshold);

                  if( !key_eligible ) continue;
              }

              key_eligible = false;
              //Try to find the entry in hash tables
              access_kernel(access_id,//for ask: relation between access_kernel and hash_kernel
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
               query::dim* d_dims,
               T_HASHTABLE* hash_table_list,
               u32 * bitmap_list,
               int bitmap_bits,
               u32* d_topks,
               u32* d_threshold,//initialized as 1, and increase gradually
               u32* d_passCount,//initialized as 0, count the number of items passing one d_threshold
               u32 num_of_max_count,
               int num_of_hot_dims,
               int hot_dim_threshold,
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

       int min, max;
       min = q.low;
       max = q.up;
       if (min > max)
         return;

       min < 1 ? min = 0 : min = d_ck[min - 1];
       max = d_ck[max];

       bool key_eligible;//for ask: what does it mean for key_eligible

       for (int i = 0; i < (max - min) / GaLG_device_THREADS_PER_BLOCK + 1; i++)
         {
     	  int tmp_id = threadIdx.x + i * GaLG_device_THREADS_PER_BLOCK + min;
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
                 		  	    &key_eligible,
                 		  	    num_of_hot_dims,
                 		  	    hot_dim_threshold);
                  //for debug
                  //if(count>=0&&blockIdx.x<128){
                  //        printf("for debug: my_passCount=%d threshold=%d  item count=%d access_id=%d\n",*my_passCount,*my_passCount/my_topk,count,access_id);
                  //   }
                  //end for debug

                   if( !key_eligible) continue;//i.e. count< thread_threshold
               }
               //for debug
               //if(count>=thread_threshold&&blockIdx.x<128){
               //       printf("for debug: my_passCount=%d threshold=%d  item count=%d access_id=%d\n",*my_passCount,*my_passCount/my_topk,count,access_id);
               // }
               //end for debug

						key_eligible = false;
						if(count< *my_threshold){
							continue;//threshold has been increased, no need to insert
						}
					   //Try to find the entry in hash tables
					   access_kernel_AT(access_id,//for ask: relation between access_kernel and hash_kernel
									 hash_table,
									 hash_table_size,
									 q,
									 count,
									 &key_eligible);

					   //for debug
					   //if(blockIdx.x<128){
						//   if(access_id==10)
						//	printf("for debug: after access_kernel_AT: CAS my_passCount=%d *my_threshold=%d  item count=%d thread_threshold=%d   my_noiih=%d access_id=%d \n",my_passCount[count],*my_threshold,count,thread_threshold,(*my_noiih),access_id);
					    //}
					   //end for debug

					   if(key_eligible){

						   updateThreshold(my_passCount,my_threshold, my_topk,count);

						   continue;
					   }

					   if(!key_eligible)
					   {
						 //Insert the key into hash table
						 //access_id and its location are packed into a packed key

//						 u32 this_passCount,this_threshold,old_passCount;
//						 do{
//							 this_passCount = *my_passCount;
//							 this_threshold = (this_passCount)/(my_topk);//access the new threshold
//							 if(thread_threshold==this_threshold){//if still remain the same threshold, do increase the passCount, else, ignore this item (do not insert into hashtable)
//								 old_passCount = atomicCAS(my_passCount,this_passCount,(this_passCount+1));
//							 }else{
//								 break;
//							 }
//						 }while(this_passCount!=old_passCount);
//
//
//
//						 if(thread_threshold!=this_threshold){
//							 continue;// if not within the same threshold, this item does not need to be inserted in the hashtable.
//						 }

						 //u32 this_threshold = (*my_threshold);

						 //for debug
						//if(threadIdx.x==0&&blockIdx.x<128){
						//	 printf("for debug: before CAS my_passCount[%d]=%d this_threshold=%d  my_threshold=%d \n",this_threshold,my_passCount[this_threshold],this_threshold,*my_threshold);
						//}
						//end for debug

//						 while(true){
//							 this_threshold = *my_threshold;
//							 if(my_passCount[this_threshold]>=my_topk){
//								 this_threshold = atomicCAS(my_threshold,this_threshold,this_threshold+1);
//
//							 }else{
//								 break;
//							 }
//						 }

						//if(blockIdx.x<128){
						//		if(access_id==4)
						//		printf("for debug: after hash_kernel_AT: CAS my_passCount=%d *my_threshold=%d  item count=%d thread_threshold=%d  this_threshold=%d my_noiih=%d access_id=%d \n",my_passCount[count],*my_threshold,count,thread_threshold,this_threshold,(*my_noiih),access_id);
						// }


						 if(count< *my_threshold){
							 continue;//threshold has been increased, no need to insert
						 }



						 hash_kernel_AT(access_id,
									 hash_table,
									 hash_table_size,
									 q,
									 count,
									 *my_threshold,
									 my_noiih,
									 overflow);
						 if(*overflow)
						 {
							return;
						 }

						 updateThreshold(my_passCount,my_threshold, my_topk,count);

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
GaLG::build_queries(vector<query>& queries, inv_table& table, vector<query::dim>& dims)
{
	for (int i = 0; i < queries.size(); ++i)
	{
	  if (queries[i].ref_table() != &table)
		throw inv_table::not_matched_exception;
	  if (table.build_status() == inv_table::builded)
		queries[i].build();
	  else if (table.build_status() == inv_table::builded_compressed)
		queries[i].build_compressed();
	  queries[i].dump(dims);
	}
}
void
GaLG::match(inv_table& table,
            vector<query>& queries,
            device_vector<data_t>& d_data,
            int hash_table_size,
            int bitmap_bits,
            int num_of_hot_dims,
            int hot_dim_threshold,
            device_vector<u32>& d_noiih)
{
	device_vector<u32> d_bitmap;
	match(table, queries,d_data,d_bitmap,hash_table_size,bitmap_bits,num_of_hot_dims,hot_dim_threshold, d_noiih);
}
void
GaLG::match(inv_table& table,
            vector<query>& queries,
            device_vector<data_t>& d_data,
            device_vector<u32>& d_bitmap,
            int hash_table_size,
            int bitmap_bits,//or for AT: for adaptiveThreshold, if bitmap_bits<0, use adaptive threshold, the absolute value of bitmap_bits is count value stored in the bitmap
            int num_of_hot_dims,
            int hot_dim_threshold,
            device_vector<u32>& d_noiih)
{
#ifdef GALG_DEBUG
	printf("match.cu version : %s\n", VERSION);
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

	u32 loop_count = 0u;
	d_noiih.resize(queries.size());
	thrust::fill(d_noiih.begin(), d_noiih.end(), 0u);
	u32 * d_noiih_p = thrust::raw_pointer_cast(d_noiih.data());

	vector<query::dim> dims;
	vector<query::dim> hot_dims;
	vector<query> hot_dims_queries;

#ifdef GALG_DEBUG
	printf("[Info]hash table size: %d.\n", hash_table_size);
#endif
	//TODO: Modify this to enable hot dim search
//	if(num_of_hot_dims)
//	{
//		for(int i = 0; i < queries.size(); ++i)
//		{
//			query q(table);
//			q.topk(queries[i].topk());
//			queries[i].split_hot_dims(q, num_of_hot_dims);
//			hot_dims_queries.push_back(q);
//		}
//		build_queries(hot_dims_queries, table, hot_dims);
//	}
//	printf("Host query size: %d.\n", queries.size());

  build_queries(queries, table, dims);

#ifdef GALG_DEBUG
  printf("[Info] dims size: %d. hot_dims size: %d.\n", dims.size(), hot_dims.size());
#endif

  //for AT: for adaptiveThreshold, enable adaptiveThreshold
  bool useAdaptiveThreshold = false;//for AT
  if(bitmap_bits<0){
	 bitmap_bits = -bitmap_bits;
	 useAdaptiveThreshold = true;
	 //for hash_table_size, still let it determine by users currently
  }
  printf("for debug: useAdaptiveThreshold=%d, bitmap_bits=%d \n",useAdaptiveThreshold,bitmap_bits);
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



#ifdef GALG_DEBUG
    printf("[info] Bitmap bits: %d, threshold:%d.\n", bitmap_bits, threshold);
	printf("[ 20%] Declaring device memory...\n");
#endif

	size_t free_ck_before,free_ck_after, free_inv_after,free_q_after,free_bitmap_after, total_m;
	cudaMemGetInfo(&free_ck_before, &total_m);
	device_vector<int> d_ck(*table.ck());
	cudaMemGetInfo(&free_ck_after, &total_m);
	int* d_ck_p = raw_pointer_cast(d_ck.data());

	device_vector<int> d_inv(*table.inv());
	int* d_inv_p = raw_pointer_cast(d_inv.data());
	cudaMemGetInfo(&free_inv_after, &total_m);

	device_vector<query::dim> d_dims(dims);
	query::dim* d_dims_p = raw_pointer_cast(d_dims.data());
	cudaMemGetInfo(&free_q_after, &total_m);

	d_bitmap.resize(bitmap_size);
	cudaMemGetInfo(&free_bitmap_after, &total_m);
	if(bitmap_size)
	{
		thrust::fill(d_bitmap.begin(), d_bitmap.end(), 0u);
	}
	u32 * d_bitmap_p = raw_pointer_cast(d_bitmap.data());

#ifdef GALG_DEBUG
	printf("d_ck size: %u\nd_inv size: %u\nquery size: %u\nbitmap size: %u.\n",
		  	free_ck_before - free_ck_after,
		  	free_ck_after - free_inv_after ,
		  	free_inv_after - free_q_after,
		  	free_q_after- free_bitmap_after);

  printf("[ 30%] Allocating device memory to tables...\n");
#endif

	data_t nulldata;
	nulldata.id = 0u;
	nulldata.aggregation = 0.0f;
	T_HASHTABLE* d_hash_table;
	data_t* d_data_table;
	d_data.clear();

	d_data.resize(queries.size()*hash_table_size);
	thrust::fill(d_data.begin(), d_data.end(), nulldata);
	d_data_table = thrust::raw_pointer_cast(d_data.data());
	d_hash_table = reinterpret_cast<T_HASHTABLE*>(d_data_table);
  
#ifdef GALG_DEBUG
  printf("[ 33%] Copying memory to symbol...\n");
#endif

  u32 h_offsets[16] = OFFSETS_TABLE_16;
  cudaCheckErrors(cudaMemcpyToSymbol(GaLG::device::offsets, h_offsets, sizeof(u32)*16, 0, cudaMemcpyHostToDevice));

#ifdef GALG_DEBUG
  printf("[ 40%] Starting match kernels...\n");
  cudaEventRecord(kernel_start);
#endif

  	bool h_overflow[1]= {false};
    bool * d_overflow;
    cudaCheckErrors(cudaMalloc((void**) &d_overflow, sizeof(bool)));

    if(!useAdaptiveThreshold) //for AT: for adaptiveThreshold, branch here
    {
	do{
		h_overflow[0] = false;
		cudaCheckErrors(cudaMemcpy(d_overflow, h_overflow, sizeof(bool), cudaMemcpyHostToDevice));
		cudaCheckErrors(cudaDeviceSynchronize());
		device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
		(table.m_size(),
		table.i_size(),
		hash_table_size,
		d_ck_p,
		d_inv_p,
		d_dims_p,
		d_hash_table,
		d_bitmap_p,
		bitmap_bits,
		threshold,
		0 /* NUM OF HOT DIM = 0 */,
		hot_dim_threshold,
		d_noiih_p,
		d_overflow);
		cudaCheckErrors(cudaDeviceSynchronize());
		cudaCheckErrors(cudaMemcpy(h_overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));
		loop_count ++;
		if(h_overflow[0])
		{
			hash_table_size += 0.1f * table.i_size();
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

#ifdef GALG_DEBUG
		printf("%d time trying to launch match kernel: %s!\n", loop_count, h_overflow[0]?"failed":"succeeded");
#endif

	} while(h_overflow[0]);
    }else{//for AT: for adaptiveThreshold, use different match method for adaptiveThreshold

    	cudaCheckErrors(cudaMemcpy(d_overflow, h_overflow, sizeof(bool), cudaMemcpyHostToDevice));
    	cudaCheckErrors(cudaDeviceSynchronize());
    	//place match_function here!

    	device_vector<u32> d_threshold;
    	d_threshold.resize(queries.size());
    	thrust::fill(d_threshold.begin(), d_threshold.end(), 1);
    	u32 * d_threshold_p = thrust::raw_pointer_cast(d_threshold.data());

    	device_vector<u32> d_passCount;
    	u32 num_of_max_count = dims.size();
    	d_passCount.resize(queries.size()*num_of_max_count);//
    	thrust::fill(d_passCount.begin(), d_passCount.end(), 0u);
    	u32 * d_passCount_p = thrust::raw_pointer_cast(d_passCount.data());

    	host_vector<u32> h_tops(queries.size());
    	for (u32 i = 0; i < queries.size(); i++)
    	 {
    	      h_tops[i] = queries[i].topk();
    	 }
    	 device_vector<u32> d_topks(h_tops);
    	 u32 * d_topks_p = thrust::raw_pointer_cast(d_topks.data());

    	device::match_AT<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
    					(table.m_size(),
    					table.i_size(),
    					hash_table_size,
    					d_ck_p,
    					d_inv_p,
    					d_dims_p,
    					d_hash_table,
    					d_bitmap_p,
    					bitmap_bits,
    					d_topks_p,
    	                d_threshold_p,//initialized as 1, and increase gradually
    	                d_passCount_p,//initialized as 0, count the number of items passing one d_threshold
    	                num_of_max_count,//number of maximum count per query
    	               0 /* NUM OF HOT DIM = 0 */,
    	               hot_dim_threshold,
    	               d_noiih_p,
    	               d_overflow);

    	//for debug
    	//printf("for debug\n");
    	//host_vector<u32> h_noiih = d_noiih;
    	//for(int i=0;i<h_noiih.size();i++){
    	//	printf("for debug: h_noiih[%d] is:%d \n",i,h_noiih[i]);
    	//}
    	//end for debug

    	cudaCheckErrors(cudaDeviceSynchronize());
    	cudaCheckErrors(cudaMemcpy(h_overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));

    }//end for AT: for adaptiveThreshold, use different match method for adaptiveThreshold

/* The following code snippet is to count the number of points in hash table
	std::vector<T_HASHTABLE> temp_data;
	temp_data.resize(hash_table_size * sizeof(T_HASHTABLE));
	cudaCheckErrors(cudaMemcpy(&temp_data.front(), d_hash_table, sizeof(T_HASHTABLE) * hash_table_size, cudaMemcpyDeviceToHost));
	u64 non_zero = 0ull;
	for(int i = 0; i < temp_data.size();++i)
	{
		if(temp_data[i] != 0ull) non_zero += 1;
	}
	temp_data.clear();
	printf("[Info] Non-zero of non-hot-dim: %llu.\n", non_zero);
 * */

	//TODO: MODIFY HOT DIM SEARCH TO ADJUST TO MULTIRANGE
	//HOT-DIM-SEARCH
//	if(num_of_hot_dims)
//	{
//		d_dims.clear();
//		device_vector<query::dim>().swap(d_dims);
//		device_vector<query::dim> d_hot_dims(hot_dims);
//		query::dim* d_hot_dims_p = raw_pointer_cast(d_hot_dims.data());
//		device::match<<<dims.size(), GaLG_device_THREADS_PER_BLOCK>>>
//			(table.m_size(),
//			table.i_size(),
//			hash_table_size,
//			d_ck_p,
//			d_inv_p,
//			d_hot_dims_p,
//			d_hash_table,
//			d_bitmap_p,
//			bitmap_bits,
//			threshold,
//			num_of_hot_dims,
//			hot_dim_threshold);
//	}
/* The following code snippet is to count the number of points in hash table
	std::vector<T_HASHTABLE> temp_data2;
	temp_data2.resize(hash_table_size * sizeof(T_HASHTABLE));
	cudaCheckErrors(cudaMemcpy(&temp_data2.front(), d_hash_table, sizeof(T_HASHTABLE) * hash_table_size, cudaMemcpyDeviceToHost));
	u64 non_zero2 = 0ull;
	for(int i = 0; i < temp_data2.size();++i)
	{
		if(temp_data2[i] != 0ull) non_zero2 += 1;
	}
	temp_data2.clear();
	printf("[Info] Non-zero of hot-dim: %llu.\n", non_zero2);
 * */
  
#ifdef GALG_DEBUG
  cudaEventRecord(kernel_stop);
  printf("[ 90%] Starting data converting......\n");
#endif

  cudaCheckErrors(cudaDeviceSynchronize());
  device::convert_to_data<<<hash_table_size*queries.size() / 1024 + 1, 1024>>>(d_hash_table,(u32)hash_table_size*queries.size());

#ifdef GALG_DEBUG
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
