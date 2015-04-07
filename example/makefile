.PHONY: build test doc clean


build:

#Building file: inv_list.cu
#Invoking: NVCC Compiler
#	src/inv_match.o:src/inv_match.cu
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "inv_list.d" "inv_list.cu"
	nvcc --device-c -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "inv_list.o" "inv_list.cu"

	nvcc --cudart shared -L../build/lib -L/lib64 --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "inv_list_bin"  inv_list.o   -lGaLG -lpthread
#Finished building target: inv_list_bin


#Building file: inv_match.cu
#Invoking: NVCC Compiler
#	src/inv_match.o:src/inv_match.cu
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "inv_match.d" "inv_match.cu"
	nvcc --device-c -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "inv_match.o" "inv_match.cu"

	nvcc --cudart shared -L../build/lib -L/lib64 --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "inv_match_bin"  inv_match.o   -lGaLG -lpthread
#Finished building target: inv_match_bin



#Building file: inv_topk.cu
#Invoking: NVCC Compiler
#	src/inv_match.o:src/inv_match.cu
	nvcc -I../src -include ../src/GaLG.h -O3 -gencode arch=compute_35,code=sm_35  -M -o "inv_topk.d" "inv_topk.cu"
	nvcc --device-c -I../src -O3 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -include ../src/GaLG.h  -x cu -o  "inv_topk.o" "inv_topk.cu"

	nvcc --cudart shared -L../build/lib -L/lib64 --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "inv_topk_bin"  inv_topk.o   -lGaLG -lpthread
#Finished building target: inv_topk_bin







clean:
	rm -rf *.o 
	rm -rf *.d
	rm inv_*_bin

