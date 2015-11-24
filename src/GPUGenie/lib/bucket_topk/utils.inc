#ifndef BUCKET_TOPK_UTILS_INC
#define BUCKET_TOPK_UTILS_INC

template <class Data, class V>
__device__ void blk_sort_inSharedMemory(Data* data_array, Data* temp_array, V vop, int size)
{
	for (short base = 1; base < size; base <<= 1) {
		int sort_round = size / blockDim.x + (size % blockDim.x != 0);
		for (int ri = 0; ri < sort_round; ri++) {
			int idx = ri * blockDim.x + threadIdx.x;
			short index = 2 * base * idx;
			if (index < size) {
				int start_idx_x = index; //threadIdx.x;
				int end_idx_x = start_idx_x + base;
				end_idx_x = end_idx_x < size ? end_idx_x : size;

				int start_idx_y = end_idx_x;
				int end_idx_y = start_idx_y + base;
				end_idx_y = end_idx_y < size ? end_idx_y : size;

				int x_pointer = start_idx_x;
				int y_pointer = start_idx_y;

				int output_ptr = x_pointer;
				while (x_pointer < end_idx_x || y_pointer < end_idx_y) {
					if (x_pointer >= end_idx_x)
						temp_array[output_ptr++] = data_array[y_pointer++];
					else if (y_pointer >= end_idx_y)
						temp_array[output_ptr++] = data_array[x_pointer++];
					else if (vop.valueOf(data_array[x_pointer]) > vop.valueOf(data_array[y_pointer]))//key operation
						temp_array[output_ptr++] = data_array[y_pointer++];
					else
						temp_array[output_ptr++] = data_array[x_pointer++];
				}
			}
		}
		__syncthreads();

		//copy back to shared_array
		int round = size / blockDim.x + (size % blockDim.x != 0);
		for (int i = 0; i < round; i++) {
			int idx = i * blockDim.x + threadIdx.x;
			if (idx < size) {
				data_array[idx] = temp_array[idx];
			}
		}
		__syncthreads();
	}
	__syncthreads();
}
#endif
