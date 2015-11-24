.PHONY: build test doc clean


build:


#Building file: general_test.cu
#Invoking: NVCC Compiler
	#nvcc -I../src -include ../src/GPUGenie.h -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52  -M -o "general_test.d" "general_test.cu"
	#nvcc --device-c  -I../src -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "general_test.o" "general_test.cu"

	#nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "general_test_bin"  general_test.o   -lGPUGenie -lpthread





#Building file: example_tweets.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GPUGenie.h -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -M -o "example_tweets.d" "example_tweets.cu"
	nvcc --device-c  -I../src -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_tweets.o" "example_tweets.cu"

	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_tweets_bin"  example_tweets.o   -lGPUGenie -lpthread


#Building file: example_sift.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GPUGenie.h -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52  -M -o "example_sift.d" "example_sift.cu"
	nvcc --device-c  -I../src -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_sift.o" "example_sift.cu"

	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_sift_bin"  example_sift.o   -lGPUGenie -lpthread






clean:
	rm -rf *.o 
	rm -rf *.d
	rm -rf *_bin

