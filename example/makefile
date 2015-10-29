.PHONY: build test doc clean


build:


#Building file: general_test.cu
#Invoking: NVCC Compiler
	#nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "general_test.d" "general_test.cu"
	#nvcc --device-c  -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "general_test.o" "general_test.cu"

	#nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "general_test_bin"  general_test.o   -lGaLG -lpthread





#Building file: example.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "example.d" "example.cu"
	nvcc --device-c  -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "example.o" "example.cu"

	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "example_bin"  example.o   -lGaLG -lpthread




#Building file: example_tweets.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "example_tweets.d" "example_tweets.cu"
	nvcc --device-c  -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "example_tweets.o" "example_tweets.cu"

	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "example_tweets_bin"  example_tweets.o   -lGaLG -lpthread


#Building file: example_adaptiveThreshold.cu
#Invoking: NVCC Compiler
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "example_adaptiveThreshold.d" "example_adaptiveThreshold.cu"
	nvcc --device-c  -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "example_adaptiveThreshold.o" "example_adaptiveThreshold.cu"

	nvcc --cudart  shared -L../lib -L/lib64 -L../build/lib --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "example_adaptiveThreshold_bin"  example_adaptiveThreshold.o   -lGaLG -lpthread






clean:
	rm -rf *.o 
	rm -rf *.d
	rm -rf *_bin

