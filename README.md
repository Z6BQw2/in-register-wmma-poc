# Accelerating Fused WMMA Operations via In-Register Data Permutation

This repository contains the source code and benchmarking suite for the paper "Accelerating Fused WMMA Operations via In-Register Data Permutation" by Tommy Morales.

**Paper pre-print available on HAL:** [Link to your HAL submission]

## Abstract

The performance of GPU kernels for AI, such as FlashAttention, is increasingly limited by on-chip data movement. A critical bottleneck in fused operations using the WMMA (Warp-Level Matrix-Multiply-Accumulate) API is the costly round-trip to shared memory required for post-matmul reductions. To address this, we propose a novel technique that performs these operations entirely within the GPU's register file. By reverse-engineering the undocumented data layout of WMMA fragments, we devised a set of in-register permutations that enable efficient intra-warp processing. This repository provides a proof-of-concept implementation and a full end-to-end validation in a FlashAttention-1 style kernel.

## Key Results Summary

| Benchmark Type                | Architecture      | Speedup vs. Shared Memory |
| ----------------------------- | ----------------- | ------------------------- |
| Micro-Benchmark (Core Op)     | NVIDIA RTX 5080   | **5.5x**                 |
| Micro-Benchmark (Core Op)     | NVIDIA H100       | **6.05x**                 |
| End-to-End FA-1 Style Kernel  | NVIDIA RTX 5080   | **1.40x**                 |
| End-to-End FA-1 Style Kernel  | NVIDIA H100   | **1.35x**                 |

## Project Structure

This project is organized to separate the CUDA kernels from the testing harnesses, all managed by a central `Makefile`.

-   `kernels.cuh`: Contains the CUDA `__global__` function definitions for both the baseline (shared memory) and the optimized (in-register) kernels.
-   `utils.cuh`: Shared helper code, including the `CUDA_CHECK` macro and the CPU reference implementation.
-   `main_correctness.cu`: A test harness to verify the numerical correctness of the GPU kernels against the CPU reference.
-   `main_latency.cu`: A minimal test harness designed to be called in a loop by `benchmark_latency.sh` for robust runtime measurements.
-   `main_power.cu`: A test harness that runs a specific kernel in an infinite loop to generate a sustained load for power measurement.
-   `main_profiling.cu`: A test harness that launches the kernels on a large grid to enable meaningful micro-architectural profiling with NVIDIA Nsight Compute.
-   `benchmark_*.sh`: Shell scripts that automate the different testing procedures.
-   `Makefile`: The main build and execution script for the entire project.

## Requirements

-   **NVIDIA GPU:** An NVIDIA GPU with Tensor Cores is required to run the code as-is.
-   **Build Tools:** A C++ compiler (`g++`) and `make`.

## Build and Test Instructions

###  Configuration

Before compiling, open the `Makefile` and edit the `ARCH` variable to match your GPU's architecture (e.g., `-arch=sm_90a` for Hopper, `-arch=sm_120` for Blackwell).

### 2. Compilation

Navigate to the project's root directory in your terminal and run:
```bash
make all
```
This will compile all four test executables: `correctness`, `latency_bench`, `power_bench`, and `profiling_bench`.

### 3. Running the Benchmarks

A master command will compile and run all tests in sequence:
```bash
make test
```
Alternatively, you can run each test individually.

#### Correctness Test
This test runs each kernel once and compares its output to the CPU reference. It's the best way to ensure the code works on your setup and that your chosen architecture flag is correct.
```bash
make run_correctness
```

#### Runtime (Latency) Benchmark
This runs a robust benchmark to measure the average execution time of the kernels, discarding warm-up runs.
```bash
make run_latency
```
You can also run the script manually to change the number of runs: `sh benchmark_latency.sh <warm-up_runs> <timed_runs>`.

#### Power Consumption Benchmark
This script measures the average power draw using `nvidia-smi` under a sustained load.
```bash
make run_power
```
You can manually adjust the test duration in the script by modifying the `TEST_DURATION` variable.

#### Micro-architectural Profiling
This script uses NVIDIA Nsight Compute (`ncu`) to collect detailed hardware metrics like occupancy and compute throughput.
```bash
make run_profiling
```

### A Note on the FlashAttention Kernel (`FlashAttentionInRegister.cu`)

The repository also includes `FlashAttentionInRegister.cu`, which provides a working example of integrating the in-register technique into a full, end-to-end FlashAttention kernel. This implementation is based on the FA-1 architecture (no V-matrix double buffering, no warp specialization). While it served to produce the 1.40x speedup result, implementing the method in more complex, state-of-the-art kernels may require additional engineering.

## License

This project is licensed under the BSD 3-Clause Licence. See the `LICENSE` file for details.

## Citation

If you use this work in your research, please cite the following paper (Currently Awaiting Moderation):
```bibtex
@misc{morales2025inregister,
      title={Accelerating Fused WMMA Operations via In-Register Data Permutation}, 
      author={Tommy Morales},
      year={2025},
      eprint={},
      archivePrefix={HAL},
      primaryClass={cs.DC}
}
```
*(Note: Update the eprint and HAL ID once available)*
