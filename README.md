# basic-cudaQ-examples

Minimal, experimental examples using **CUDA-Q in C++**, intended for HPC-style environments.

This repository exists primarily to explore how CUDA-Q behaves **below the Python abstraction layer**, including integration with native CUDA kernels.

---

## Why C++ only?

The system this was developed on uses **NVIDIA Volta GPUs (V100 16GB)**. Recent versions of CUDA-Q [no longer supported Volta](https://github.com/NVIDIA/cuda-quantum/issues/3706#issuecomment-3717374073), and due to my own laziness to downgrade (and I already have CUDAQ 0.11 w/ nvq++ compiler on my machine anyway), this effectively rules out Python-based CUDA-Q for this repository.

As a result, all CUDA-Q examples here are written in C++, which allows us to do more finnicky stuff with NVIDIA's CUDA ecosystem.

(Qiskit Aer still runs fine on Volta, which is why Python examples exist in the Qiskit repo, but not here.)

---

## Repository structure

Files are prefixed numerically to indicate intent:

- `0_*.cpp`  
  Vanilla CUDA-Q examples (quantum-only)

- `1_*.cpp`  
  CUDA-Q examples integrated with native CUDA kernels, [demonstrating hybrid classical–quantum execution within a single executable](https://nvidia.github.io/cuda-quantum/latest/using/integration/cuda_gpu.html)

CUDA kernels live in `.cu` files and are compiled with `nvcc`; CUDA-Q code is compiled with `nvq++`. For more information on CUDA & CUDA-Q integration, please consult the `makefile`.

---

## Building the project

To compile `.cpp` CUDA-Q files, simply:

```bash
nvq++ 0_deutsch_jozsa_phase_oracle.cpp -o deutsch_jozsa_phase_oracle
```

To compile the CUDA and CUDA-Q files as an integration project, simply run the `makefile`

```bash
make clean && make -j$(nproc)
```

---

## References
	•	https://arxiv.org/pdf/1907.09415
	•	NVIDIA CUDAQ Documentation
