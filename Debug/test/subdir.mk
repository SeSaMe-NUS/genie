################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../test/test_match.cu \
../test/test_topk.cu 

CC_SRCS += \
../test/test_inv_list.cc \
../test/test_inv_table.cc \
../test/test_raw_data.cc 

CU_DEPS += \
./test/test_match.d \
./test/test_topk.d 

OBJS += \
./test/test_inv_list.o \
./test/test_inv_table.o \
./test/test_match.o \
./test/test_raw_data.o \
./test/test_topk.o 

CC_DEPS += \
./test/test_inv_list.d \
./test/test_inv_table.d \
./test/test_raw_data.d 


# Each subdirectory must supply rules for building sources it contributes
test/%.o: ../test/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

test/%.o: ../test/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


