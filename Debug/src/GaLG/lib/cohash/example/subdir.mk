################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/GaLG/lib/cohash/example/img_tga.cpp 

CU_SRCS += \
../src/GaLG/lib/cohash/example/main.cu \
../src/GaLG/lib/cohash/example/test_cu_robin_hood_hash.cu 

CU_DEPS += \
./src/GaLG/lib/cohash/example/main.d \
./src/GaLG/lib/cohash/example/test_cu_robin_hood_hash.d 

OBJS += \
./src/GaLG/lib/cohash/example/img_tga.o \
./src/GaLG/lib/cohash/example/main.o \
./src/GaLG/lib/cohash/example/test_cu_robin_hood_hash.o 

CPP_DEPS += \
./src/GaLG/lib/cohash/example/img_tga.d 


# Each subdirectory must supply rules for building sources it contributes
src/GaLG/lib/cohash/example/%.o: ../src/GaLG/lib/cohash/example/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG/lib/cohash/example" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/GaLG/lib/cohash/example/%.o: ../src/GaLG/lib/cohash/example/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/GaLG/lib/cohash/example" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


