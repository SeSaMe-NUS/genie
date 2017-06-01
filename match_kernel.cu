#include <mpi.h>
#include <unistd.h>

void match_AT_DY(int m_size, int i_size, int hash_table_size,
		int* d_inv, query::dim* d_dims,
		T_HASHTABLE* hash_table_list, u32 * bitmap_list, int bitmap_bits,
		u32* d_topks, u32* d_threshold, //initialized as 1, and increase gradually
		u32* d_passCount, //initialized as 0, count the number of items passing one d_threshold
		u32 num_of_max_count, u32 * noiih, bool * overflow, unsigned int shift_bits_subsequence,
		u32 *aux_threshold, // NEW INTRODUCED VAR
		int dim_offset) // NEW INTRODUCED VAR
{
	if (m_size == 0 || i_size == 0)
		return;
	query::dim& q = d_dims[blockIdx.x + dim_offset];
	int query_index = q.query;
	u32* my_noiih = &noiih[query_index];
	u32* my_threshold = &d_threshold[query_index];
	//__shared__ u32 my_threshold;
	//my_threshold = d_threshold[query_index];
	//__syncthreads();
#ifdef USE_DYNAMIC
	// update main array from aux array
	u32 *my_aux_threshold = &aux_threshold[query_index];
	*my_threshold = *my_aux_threshold;
#endif
	u32* my_passCount = &d_passCount[query_index * num_of_max_count];         //
	u32 my_topk = d_topks[query_index];                //for AT

	T_HASHTABLE* hash_table = &hash_table_list[query_index * hash_table_size];
	u32 * bitmap;
	if (bitmap_bits)
		bitmap = &bitmap_list[query_index * (i_size / (32 / bitmap_bits) + 1)];
	u32 access_id;
	int min, max, order;
	if(q.start_pos >= q.end_pos)
		return;

	min = q.start_pos;
	max = q.end_pos;
	order = q.order;
	bool key_eligible;                //
	bool pass_threshold;    //to determine whether pass the check of my_theshold

	for (int i = 0; i < (max - min - 1) / GPUGenie_device_THREADS_PER_BLOCK + 1; ++i)
	{

		int tmp_id = threadIdx.x + i * GPUGenie_device_THREADS_PER_BLOCK + min;
		if (tmp_id < max)
		{
			u32 count = 0;                //for AT
			access_id = d_inv[tmp_id];

            if(shift_bits_subsequence != 0)
            {
                int __offset = access_id & (((unsigned int)1<<shift_bits_subsequence) - 1);
                int __new_offset = __offset - order;
                if(__new_offset >= 0)
                {
                    access_id = access_id - __offset + __new_offset;
                }
                else
                    continue;
            }

			u32 thread_threshold = *my_threshold;
			//u32 thread_threshold = my_threshold;
			if (bitmap_bits)
			{

				key_eligible = false;
				//all count are store in the bitmap, and access the count
				count = bitmap_kernel_AT(access_id, bitmap, bitmap_bits,
						thread_threshold, &key_eligible);

				if (!key_eligible)
					continue;                //i.e. count< thread_threshold
			}

			key_eligible = false;
			if (count < *my_threshold)
			//if (count < my_threshold)
			{
				continue;      //threshold has been increased, no need to insert
			}

			//Try to find the entry in hash tables
			access_kernel_AT(
					access_id,               
					hash_table, hash_table_size, q, count, &key_eligible,
					my_threshold, &pass_threshold);
					//&my_threshold, &pass_threshold);

			if (key_eligible)
			{
				if (pass_threshold)
				{
					updateThreshold(my_passCount, my_threshold, my_topk, count);
					//updateThreshold(my_passCount, &my_threshold, my_topk, count);
#ifdef USE_DYNAMIC
					// also update aux array
					//updateThreshold(my_passCount, my_aux_threshold, my_topk, count);
#endif
				}

				continue;
			}

			if (!key_eligible)
			{
				//Insert the key into hash table
				//access_id and its location are packed into a packed key

				if (count < *my_threshold)
				//if (count < my_threshold)
				{
					continue;//threshold has been increased, no need to insert
				}

				hash_kernel_AT(access_id, hash_table, hash_table_size, q, count,
						my_threshold, my_noiih, overflow, &pass_threshold);
						//&my_threshold, my_noiih, overflow, &pass_threshold);
				if (*overflow)
				{

					return;
				}
				if (pass_threshold)
				{
					updateThreshold(my_passCount, my_threshold, my_topk, count);
					//updateThreshold(my_passCount, &my_threshold, my_topk, count);
#ifdef USE_DYNAMIC
					// also update aux array
					//updateThreshold(my_passCount, my_aux_threshold, my_topk, count);
#endif
				}
			}

		}
	}
#ifdef USE_DYNAMIC	
	// update main threshold to aux threshold after processing a query
	*my_aux_threshold = *my_threshold;
#endif
}
