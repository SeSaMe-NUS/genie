################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../test/parser/test_csv.cc 

OBJS += \
./test/parser/test_csv.o 

CC_DEPS += \
./test/parser/test_csv.d 


# Each subdirectory must supply rules for building sources it contributes
test/parser/%.o: ../test/parser/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "test/parser" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


