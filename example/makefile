#.PHONY: build test doc clean
#
#
build:
#
#
###Building file: example_tweets.cu
###Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GPUGenie.h -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -M -o "example_tweets.d" "example_tweets.cu"
	nvcc --device-c  -I../src -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_tweets.o" "example_tweets.cu"
	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_tweets_bin"  example_tweets.o   -lGPUGenie -lpthread
##
##Building file: example_tweets_binary.cu
##Invoking: NVCC Compiler
#	nvcc -I../src -include ../src/GPUGenie.h -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -M -o "example_tweets_binary.d" "example_tweets_binary.cu"
#	nvcc --device-c  -I../src -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_tweets_binary.o" "example_tweets_binary.cu"
#	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_tweets_binary_bin"  example_tweets_binary.o   -lGPUGenie -lpthread 
###


#Building file: example_sift.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GPUGenie.h -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52  -M -o "example_sift.d" "example_sift.cu"
	nvcc --device-c  -I../src -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_sift.o" "example_sift.cu"
	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_sift_bin"  example_sift.o   -lGPUGenie -lpthread 

#Building file: example_sift_binary.cu
#Invoking: NVCC Compiler
#	nvcc -I../src -include ../src/GPUGenie.h -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52  -M -o "example_sift_binary.d" "example_sift_binary.cu"
#	nvcc --device-c  -I../src -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_sift_binary.o" "example_sift_binary.cu"
#	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_sift_binary_bin"  example_sift_binary.o   -lGPUGenie -lpthread 

#Building file: example_read_table_from_file.cu
#Invoking: NVCC Compiler
#	nvcc -I../src -include ../src/GPUGenie.h -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52  -M -o "example_table_from_file.d" "example_table_from_file.cu"
#	nvcc --device-c  -I../src -O3  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -include ../src/GPUGenie.h  -x cu -o  "example_table_from_file.o" "example_table_from_file.cu"
#	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -link -o  "example_table_from_file_bin"  example_table_from_file.o   -lGPUGenie -lpthread 
#
#
#
##Building file: convert_csv2binary.cu
##Invoking: NVCC Compiler
##	nvcc -I../src -include ../src/GPUGenie.h -L../build/lib convert_csv2binary.cu -o convert -L../lib

clean:
	rm -rf *.o 
	rm -rf *.d
	rm -rf *_bin

