/*  Reference: https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html#defining-kernels + https://arxiv.org/pdf/1907.09415

    The Deutsch-Jozsa circuit is defined by an n-qubit system, which each qubits were initialized by a Hadamard gate, before passing through a 
    phase-oracle, and applying hadamard once more on each qubit before being measured.

    By the way we're doing this on cpp because I am too lazy to download the older version of cudaQ for python (since I am using NVIDIA V100 as the 
    main GPU).

    The difference between this and deutsch_jozsa_phase_oracle is the fact that this code also runs a CUDA code. Python could never.

    Reference for CUDA linking: https://nvidia.github.io/cuda-quantum/latest/using/integration/cuda_gpu.html
*/

#include <cuda_runtime.h>
#include <cudaq.h> 
#include "1_test_gpu_kernel.h"  // For CUDA
#include <iostream>

// CUDA check
#define CHECK_CUDA(expr)                                                \
    do {                                                                \
        cudaError_t _err = (expr);                                      \
        if (_err != cudaSuccess) {                                      \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)


// Defining the oracle
__qpu__ void constant_zero_oracle(cudaq::qubit &q){  // Btw this is done per-qubit, so you need to loop it on the main circuit constructor
    // Do nothing
}

// Main circuit constructor
__qpu__ void deutsch_jozsa_phase_oracle(int N, unsigned int mask){
    // Initial state
    cudaq::qvector q(N);  // Initial qubit count 
    cudaq::h(q);  // Rather than interating, cudaQ applies the hadamard to all of the qubit

    // Loop to apply oracle w.r.t mask (btw this is phase oracle)
    for (int i = 0; i < q.size(); i++){
        if (mask & (1u << i)){
            z(q[i]);  // Apply z-gate w/ an assumed bitmask
        }  
        // constant_zero_oracle(q[i]);  
    }

    cudaq::h(q);  // Final hadamard
}


int main() {

    // GPU code
    std::cout << "Running CUDA code..." << std::endl;

    int *d_out;
    int h_out = -1;
    int value = 67;  // https://youtu.be/v0NDDoNRtQ8?si=1b3NpHEbFp-VOEc3&t=23

    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));
    launch_basic_kernel(d_out, 67);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Host sees: " << h_out << std::endl;
    CHECK_CUDA(cudaFree(d_out));

    // CUDA-Q code
    std::cout << "Running CUDA-Q code..." << std::endl;

    int N = 5;  // Num of qubits

    // Bitmask for the balanced phase oracle
    unsigned int balanced_mask = (1u << 0) | (1u << 1) | (1u << 2);

    // Sampling the circuit
    auto result = cudaq::sample(deutsch_jozsa_phase_oracle, N, balanced_mask);
    result.dump();

    return 0;
 
}