################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/GaLG/match.cu \
../src/GaLG/topk.cu 

CC_SRCS += \
../src/GaLG/inv_list.cc \
../src/GaLG/inv_table.cc \
../src/GaLG/query.cc \
../src/GaLG/raw_data.cc 

CU_DEPS += \
./src/GaLG/match.d \
./src/GaLG/topk.d 

OBJS += \
./src/GaLG/inv_list.o \
./src/GaLG/inv_table.o \
./src/GaLG/match.o \
./src/GaLG/query.o \
./src/GaLG/raw_data.o \
./src/GaLG/topk.o 

CC_DEPS += \
./src/GaLG/inv_list.d \
./src/GaLG/inv_table.d \
./src/GaLG/query.d \
./src/GaLG/raw_data.d 


# Each subdirectory must supply rules for building sources it contributes
src/GaLG/%.o: ../src/GaLG/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/GaLG/%.o: ../src/GaLG/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


