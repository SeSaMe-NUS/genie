build:
	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_1_bin test_1.cu -lGPUGenie -lpthread

clean:
	rm -rf *_bin
