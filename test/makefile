build:
#	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_1_bin test_1.cu -lGPUGenie -lpthread
#	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_2_bin test_2.cu -lGPUGenie -lpthread
#	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_3_bin test_3.cu -lGPUGenie -lpthread
#	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_4_bin test_4.cu -lGPUGenie -lpthread
#	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_5_bin test_5.cu -lGPUGenie -lpthread
	nvcc -I../src -L../lib -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -o test_6_bin test_6.cu -lGPUGenie -lpthread

clean:
	rm -rf GPUGENIE_LOG*
	rm -rf *_bin
