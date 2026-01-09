CUDA_HOME ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
NVQPP ?= nvq++
NVCC ?= nvcc

NVCC_FLAGS = -std=c++20 -O3 -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70
NVQPP_FLAGS = -std=c++20 -O3

LIBS = -I${CUDA_HOME}/include/ -L$(CUDA_HOME)/lib64 -lcudart

all: test_run

src/1_test_gpu_kernel.o: src/1_test_gpu_kernel.cu src/1_test_gpu_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c src/1_test_gpu_kernel.cu -o src/1_test_gpu_kernel.o 

src/1_cuda_cudaq_quantum_kernel.o: src/1_cuda_cudaq_quantum_kernel.cpp src/1_test_gpu_kernel.h
	$(NVQPP) $(NVQPP_FLAGS) -c src/1_cuda_cudaq_quantum_kernel.cpp -o src/1_cuda_cudaq_quantum_kernel.o

test_run: src/1_test_gpu_kernel.o src/1_cuda_cudaq_quantum_kernel.o
	$(NVQPP) src/1_test_gpu_kernel.o src/1_cuda_cudaq_quantum_kernel.o -o test_run $(LIBS)

.PHONY: all clean
clean:
	rm -f src/*.o test_run src/*.tmp 