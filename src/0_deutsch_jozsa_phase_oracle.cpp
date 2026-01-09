/*  Reference: https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html#defining-kernels + https://arxiv.org/pdf/1907.09415

    The Deutsch-Jozsa circuit is defined by an n-qubit system, which each qubits were initialized by a Hadamard gate, before passing through a 
    phase-oracle, and applying hadamard once more on each qubit before being measured.

    By the way we're doing this on cpp because I am too lazy to download the older version of cudaQ for python (since I am using NVIDIA V100 as the 
    main GPU).
*/

#include <cudaq.h> 
#include <iostream>


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

    int N = 5;  // Num of qubits

    // Bitmask for the balanced phase oracle
    unsigned int balanced_mask = (1u << 0) | (1u << 1) | (1u << 2);

    // Sampling the circuit
    auto result = cudaq::sample(deutsch_jozsa_phase_oracle, N, balanced_mask);
    result.dump();
    return 0;
 
}