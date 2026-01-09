// Ultra-basic "just get this run" CUDA kernel

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void basic_kernel(int *out, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        out[0] = value;
        printf("GPU wrote value: %d\n", value);
    }
}

extern "C"
void launch_basic_kernel(int *d_out, int value) {
    // 1 block, 1 thread: minimal sanity kernel + makes it easier for everyone to chew
    basic_kernel<<<1, 1>>>(d_out, value);
}