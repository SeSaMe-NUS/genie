/** \file match_common.h
 *  \brief Basic utility functions to be used in matching kernels
 *
 * This file is not a standalone translation unit. Instead, it is directly included in sources of matching functions.
 * The reason is that nvcc treats __forceinline__ and __inline__ (in release mode only) functions as static = does
 * not export symbols in the library, only inlines them, similarly to C.
 *
 * Including this file allows two separate translation units to have a copy of these functions and inline them.
 */

#ifndef GENIE_MATCH_DEVICE_UTILS_H
#define GENIE_MATCH_DEVICE_UTILS_H

#include <genie/query/query.h>
#include "match_common.h"

namespace genie
{
namespace matching
{

const T_AGE    MAX_AGE = 16u;
const uint32_t KEY_TYPE_BITS = 28u;
const uint32_t KEY_TYPE_MASK = u32(u64((1ull) << KEY_TYPE_BITS) - 1u);
const uint32_t ATTACH_ID_TYPE_BITS = 32u;
const uint32_t ATTACH_ID_TYPE_MASK = u32(u64((1ull) << ATTACH_ID_TYPE_BITS) - 1ul);
const uint32_t KEY_TYPE_INIT_AGE = 1u;
const uint32_t KEY_TYPE_NULL_AGE = 0u;

static const uint32_t h_offsets[] =
    {      0u, 3949349u, 8984219u, 9805709u, 7732727u, 1046459u, 9883879u, 4889399u,
     2914183u, 3503623u, 1734349u, 8860463u, 1326319u, 1613597u, 8604269u, 9647369u};

static __device__  __constant__ u32 d_offsets[16];

__forceinline__  __host__  __device__ T_KEY get_key_pos(T_HASHTABLE key)
{
    return key & KEY_TYPE_MASK;
}

__forceinline__  __host__  __device__ T_AGE get_key_age(T_HASHTABLE key)
{
    return ((key) >> (ATTACH_ID_TYPE_BITS + KEY_TYPE_BITS));
}

__host__ __forceinline__ __device__
u32 get_key_attach_id(T_HASHTABLE key) //to get the count of one item
{
    return ((key) >> (KEY_TYPE_BITS)) & ATTACH_ID_TYPE_MASK;
}
__host__ __forceinline__ __device__
T_HASHTABLE pack_key_pos(T_KEY p)
{
    return ((p) & KEY_TYPE_MASK);
}
__host__ __forceinline__ __device__
T_HASHTABLE pack_key_pos_and_attach_id_and_age(T_KEY p, u32 i, T_AGE a)
{
    return u64(
            ((u64(a) << (ATTACH_ID_TYPE_BITS + KEY_TYPE_BITS)))
                    + ((u64(i) & ATTACH_ID_TYPE_MASK) << (KEY_TYPE_BITS))
                    + u64(p & KEY_TYPE_MASK));
}

__forceinline__  __device__ u32 hash(T_KEY key, T_AGE age,
        int hash_table_size)
{
    return (d_offsets[age] + key) % hash_table_size;
}

__forceinline__ __device__ __host__
void print_binary(char * b, u32 data)
{
    for (int i = 31; i >= 0; i--)
        b[31 - i] = ((data >> i) & 1) == 1 ? '1' : '0';
    b[32] = '\0';
}

__forceinline__ __device__ __host__ u32
get_count(u32 data, int offset, int bits)
{
    return (data >> offset) & ((1u << bits) - 1u);
}

__forceinline__ __device__ __host__ u32
pack_count(u32 data, int offset, int bits, u32 count)
{
    u32 r;
    r = data & (~(((1u << bits) - 1u) << offset));
    r |= (count << offset);
    return r;
}
__forceinline__ __device__
void access_kernel(u32 id, T_HASHTABLE* htable, int hash_table_size,
        GPUGenie::query::dim& q, bool * key_found)
{
    u32 location;
    T_HASHTABLE out_key, new_key;
    T_AGE age = KEY_TYPE_NULL_AGE;

    location = hash(id, age, hash_table_size);

    while (1)
    {
        out_key = htable[location];

        if (get_key_pos(out_key)
                == id && get_key_age(out_key) != KEY_TYPE_NULL_AGE
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
            }
        }
        else
        {
            break;
        }
    }

    while (age < MAX_AGE)
    {
        age++;
        location = hash(id, age, hash_table_size);
        out_key = htable[location];

        if (get_key_pos(out_key)
                == id && get_key_age(out_key) != KEY_TYPE_NULL_AGE
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
            }
            else
            {
                age --;
                continue;
            }
        }
    }
        //Entry not found.
    *key_found = 0;
}

//for AT: for adaptiveThreshold
__device__ __forceinline__ void
access_kernel_AT(u32 id, T_HASHTABLE* htable, int hash_table_size,
        GPUGenie::query::dim& q, u32 count, bool * key_found, u32* my_threshold,
        bool * pass_threshold // if the count smaller that my_threshold, do not insert
        )
{
    u32 location;
    T_HASHTABLE out_key, new_key;
    T_AGE age = KEY_TYPE_NULL_AGE;

    location = hash(id, age, hash_table_size);
    while (1)
    {
        out_key = htable[location];

        if (get_key_pos(out_key)
                == id && get_key_age(out_key) != KEY_TYPE_NULL_AGE
                && get_key_age(out_key) < MAX_AGE)
        {
            u32 attach_id = get_key_attach_id(out_key); //for AT: for adaptiveThreshold
            float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold
            float value_plus = count;//for AT: for adaptiveThreshold
            if(value_plus <value_1)
            {                //for AT: for adaptiveThreshold
                *pass_threshold = true;// still need to update the my_threshold and passCount
                *key_found =true;//already find the key, but do not update
                return;
            }
            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
                    *reinterpret_cast<u32*>(&value_plus),
                    get_key_age(out_key));
            if(value_plus<*my_threshold)
            {
                *pass_threshold = false; // if my_threshold is updated, no need to update hash_table and threshold
                *key_found =true;//already find the key, but do not update
                return;
            }
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {   *pass_threshold = true; //high possible that pass the threshold, must update the threshold
                *key_found =true;
                return;
            }
        }
        else
        {
            break;
        }
    }

    while (age < MAX_AGE)
    {
        age++;
        location = hash(id, age, hash_table_size);
        out_key = htable[location];

        if (get_key_pos(out_key)
                == id && get_key_age(out_key) != KEY_TYPE_NULL_AGE
                && get_key_age(out_key) < MAX_AGE)
        {
            u32 attach_id = get_key_attach_id(out_key); //for AT: for adaptiveThreshold

            float value_1 = *reinterpret_cast<float*>(&attach_id);//for AT: for adaptiveThreshold  //for improve: update here for weighted distance
            float value_plus = count;//for AT: for adaptiveThreshold
            if(value_plus <value_1)
            {              //for AT: for adaptiveThreshold
                *pass_threshold = true;// still need to update the my_threshold and passCount
                *key_found =true;//already find the key, but do not update
                return;
            }

            new_key = pack_key_pos_and_attach_id_and_age(get_key_pos(out_key),
                    *reinterpret_cast<u32*>(&value_plus), //for impprove:update here for weighted distance
                    get_key_age(out_key));
            if(value_plus<*my_threshold)
            {
                *pass_threshold = false; // if my_threshold is updated, no need to update hash_table and threshold
                *key_found =true;//already find the key, but do not update
                return;
            }
            if(atomicCAS(&htable[location], out_key, new_key) == out_key)
            {
                *pass_threshold = true;
                *key_found =true;
                return;
            }
            else
            {
                age --;
                continue;
            }
        }
    }

    *key_found = false;
    //key not found, no need to update my_threshold
    *pass_threshold = false;
}

//for AT: for countHeap (with adaptiveThreshold)
__device__ __forceinline__ void
hash_kernel_AT(
        u32 id,        
        T_HASHTABLE* htable, int hash_table_size, GPUGenie::query::dim& q, u32 count,
        u32* my_threshold, //for AT: for adaptiveThreshold, if the count is smaller than my_threshold, this item is also expired in the hashTable
        u32 * my_noiih, bool * overflow, bool* pass_threshold)
{
    u32 location;
    T_HASHTABLE evicted_key, peek_key;
    T_AGE age = KEY_TYPE_NULL_AGE;
    float count_value = count;
    T_HASHTABLE key = pack_key_pos_and_attach_id_and_age(id,
    //*reinterpret_cast<u32*>(&(q.weight)),//for AT: for adaptiveThreshold
            *reinterpret_cast<u32*>(&count_value), KEY_TYPE_INIT_AGE);
    //Loop until MAX_AGE
    while (age < MAX_AGE)
    {

        //evict key at current age-location
        //Update it if the to-be-inserted key is of a larger age
        u32 key_attach_id = get_key_attach_id(key); //for AT: for daptiveThreshold for ask: what is old_value_1, and what is old_value_2
        float key_value = *reinterpret_cast<float*>(&key_attach_id);
        if (key_value < *my_threshold)
        {           //no need to update
            if (get_key_pos(key) == id)
            {
                *pass_threshold = false; //  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
            }
            else
            {
                *pass_threshold = true; //the id has been inserted into hashtable, this key_attach_id is from the evicted_key
            }
            return;
        }

        location = hash(get_key_pos(key), age, hash_table_size);
        while (1)
        {
            if (*my_noiih > hash_table_size)
            {
                *overflow = true;
                return;
            }

            peek_key = htable[location];
            u32 peek_key_attach_id = get_key_attach_id(peek_key); //for AT: for adaptiveThreshold
            float peek_key_value =
                    *reinterpret_cast<float*>(&peek_key_attach_id);
            if (get_key_pos(peek_key) == get_key_pos(key)
                    && get_key_age(peek_key) != 0u) //even previously key_eligible ==0, the key may be inserted by other threads
            {

                //float old_value_plus = (old_value_1>old_value_2)? (*reinterpret_cast<float*>(&old_value_1)) : (*reinterpret_cast<float*>(&old_value_2));//for AT: for adaptiveThreshold

                //float old_value_plus = (old_value_1>old_value_2)? (old_value_1) : (old_value_2);//for AT: for adaptiveThreshold
                if (key_value < peek_key_value)
                {           //no need to update
                    *pass_threshold = true; // still need to update the my_threshold and passCount
                    return;
                }

                T_HASHTABLE new_key = pack_key_pos_and_attach_id_and_age(
                        get_key_pos(peek_key),
                        *reinterpret_cast<u32*>(&key_value), //for improve: update here for weighted distance
                        get_key_age(peek_key));

                if (key_value < *my_threshold)
                {           //no need to update
                    if (get_key_pos(key) == id)
                    {
                        *pass_threshold = false; //  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
                    }
                    else
                    {
                        *pass_threshold = true; //the id has been inserted into hashtable, this key_attach_id is from the evicted_key
                    }
                    return;
                }
                if (atomicCAS(&htable[location], peek_key, new_key) == peek_key)
                {

                    *pass_threshold = true; //after updat the hashtable, increase the pass_count and my_threshold
                    return;
                }
                else
                {
                    continue;
                }
            }

            if ((get_key_age(peek_key) < get_key_age(key) //if this location with smaller age (inclusive empty location, i.e. age 0)
                    || (get_key_age(peek_key) != KEY_TYPE_NULL_AGE
                            && peek_key_value < *my_threshold)) //for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
            //this item is also expired in the hashTable,
            )
            {
                if (key_value < *my_threshold)
                {                                           //no need to update
                    if (get_key_pos(key) == id)
                    {
                        *pass_threshold = false; //  if the item is expired because my_threshold is increased, no need to update hash_table and threshold by this data item
                    }
                    else
                    {
                        *pass_threshold = true; //the id has been inserted into hashtable, this key_attach_id is from the evicted_key
                    }
                    return;
                }

                evicted_key = atomicCAS(&htable[location], peek_key, key);

                if (evicted_key != peek_key)
                    continue;

                if ((get_key_age(evicted_key) > 0u) //if this not an empty location
                )
                {
                    if (peek_key_value < *my_threshold)
                    { // for AT: for adaptiveThreshold, if the count is smaller than my_threshold,
                      //this item is also expired in the hashTable,
                        *pass_threshold = true; //after updating the hashtable, increase the pass_count and my_threshold
                        return;
                    }

                    key = evicted_key;
                    age = get_key_age(evicted_key);

                    break;
                }
                else//if get_key_age(evicted_key) == 0, this is empty insertion, nothing need to do
                {

                    if (*my_noiih >= hash_table_size)
                    {
                        *overflow = true;
                        atomicAdd(my_noiih, 1u);// this will not affect the performance very much
                        return;
                    }
                    else
                    {
                        atomicAdd(my_noiih, 1u);// this will not affect the performance very much
                    }
                    *pass_threshold = true; //after updating the hashtable, increase the pass_count and my_threshold

                    return;             //finish insertion for empty location
                }
            }
            else
            {
                age++;
                key = pack_key_pos_and_attach_id_and_age(get_key_pos(key),
                        get_key_attach_id(key), age);
                break;
            }
        }

    }
    *overflow = true;
    *pass_threshold = true;
    return;
}

//for AT: for adaptiveThreshold, this is function for bitmap
__device__ __forceinline__ u32
bitmap_kernel_AT(u32 access_id, u32 * bitmap, int bits, int my_threshold,
        bool * key_eligible)
{
    u32 value, count = 0, new_value;
    int offset;

    // This loop attemps to increase the count at the corresponding location in the bitmap array (this array counts
    // the docIDs masked by first "bits" bits) until the increase is successfull, sincemany threads may be accessing
    // this bitmap array in parallel.
    while (1)
    {
        value = bitmap[access_id / (32 / bits)]; // Current value
        offset = (access_id % (32 / bits)) * bits;
        count = get_count(value, offset, bits);
        count = count + 1; //always maintain the count in bitmap//for improve: change here for weighted distance
        *key_eligible = count >= my_threshold;
        new_value = pack_count(value, offset, bits, count);
        if (atomicCAS(&bitmap[access_id / (32 / bits)], value, new_value)
                == value)
            break;
    }
    return count; //fail to access the count

}

__device__ __forceinline__ void
updateThreshold(u32* my_passCount, u32* my_threshold,
        u32 my_topk, u32 count)
{
    if (count < *my_threshold)
    {
        return;                //threshold has been increased, no need to update
    }
    atomicAdd(&my_passCount[count], 1);                //successfully update

    u32 this_threshold = (*my_threshold);

    while (true)
    {
        this_threshold = *my_threshold;
        if (my_passCount[this_threshold] >= my_topk)
        {
            this_threshold = atomicCAS(my_threshold, this_threshold,
                    this_threshold + 1);
        }
        else
        {
            break;
        }
    }
}

} // namespace matching

} // namespace genie

#endif
