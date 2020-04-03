################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/asebal.cu \
../src/candidate.cu \
../src/esaSebal.cu \
../src/filter.cu \
../src/landsat.cu \
../src/main.cu \
../src/products.cu \
../src/rah_cycle.cu 

CPP_SRCS += \
../src/parameters.cpp \
../src/pixel_reader.cpp \
../src/utils.cpp 

OBJS += \
./src/asebal.o \
./src/candidate.o \
./src/esaSebal.o \
./src/filter.o \
./src/landsat.o \
./src/main.o \
./src/parameters.o \
./src/pixel_reader.o \
./src/products.o \
./src/rah_cycle.o \
./src/utils.o 

CU_DEPS += \
./src/asebal.d \
./src/candidate.d \
./src/esaSebal.d \
./src/filter.d \
./src/landsat.d \
./src/main.d \
./src/products.d \
./src/rah_cycle.d 

CPP_DEPS += \
./src/parameters.d \
./src/pixel_reader.d \
./src/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_53,code=sm_53  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -G -g -lineinfo -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_53,code=compute_53 -gencode arch=compute_53,code=sm_53  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_53,code=sm_53  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -G -g -lineinfo -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


