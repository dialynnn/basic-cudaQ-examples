CUDA_HOME ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
NVQPP ?= nvq++
NVCC ?= nvcc

CXXFLAGS_NVCC = -std=c++20 -O3 -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70
CXXFLAGS_NVQ  = -std=c++20 -O3

LIBS = -I${CUDA_HOME}/include/ -L$(CUDA_HOME)/lib64 -lcudart

all: test_run

1_test_gpu_kernel.cu.o: 1_test_gpu_kernel.cu 1_test_gpu_kernel.h
	$(NVCC) $(CXXFLAGS_NVCC) -c 1_test_gpu_kernel.cu -o 1_test_gpu_kernel.cu.o 

1_cuda_cudaq_quantum_kernel.cpp.o: 1_cuda_cudaq_quantum_kernel.cpp 1_test_gpu_kernel.h
	$(NVQPP) $(CXXFLAGS_NVQ) -c 1_cuda_cudaq_quantum_kernel.cpp -o 1_cuda_cudaq_quantum_kernel.cpp.o

test_run: 1_test_gpu_kernel.cu.o 1_cuda_cudaq_quantum_kernel.cpp.o
	$(NVQPP) 1_test_gpu_kernel.cu.o 1_cuda_cudaq_quantum_kernel.cpp.o -o test_run $(LIBS)

clean:
	rm -f *.o test_run *.tmp