################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/GaLG/lib/libcsv/libcsv.c 

OBJS += \
./src/GaLG/lib/libcsv/libcsv.o 

C_DEPS += \
./src/GaLG/lib/libcsv/libcsv.d 


# Each subdirectory must supply rules for building sources it contributes
src/GaLG/lib/libcsv/%.o: ../src/GaLG/lib/libcsv/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG/lib/libcsv" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


