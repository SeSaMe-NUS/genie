################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/GaLG/parser/csv.cc 

OBJS += \
./src/GaLG/parser/csv.o 

CC_DEPS += \
./src/GaLG/parser/csv.d 


# Each subdirectory must supply rules for building sources it contributes
src/GaLG/parser/%.o: ../src/GaLG/parser/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG/parser" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


