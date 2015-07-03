################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../example/general_test.o \
../example/inv_list.o \
../example/inv_match.o \
../example/inv_topk.o 

CU_SRCS += \
../example/general_test.cu \
../example/inv_list.cu \
../example/inv_match.cu \
../example/inv_topk.cu \
../example/raw_data.cu 

CU_DEPS += \
./example/general_test.d \
./example/inv_list.d \
./example/inv_match.d \
./example/inv_topk.d \
./example/raw_data.d 

OBJS += \
./example/general_test.o \
./example/inv_list.o \
./example/inv_match.o \
./example/inv_topk.o \
./example/raw_data.o 


# Each subdirectory must supply rules for building sources it contributes
example/%.o: ../example/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "example" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


