#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "kernels.cuh" 

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Ref CPU (identique à ton code)
void cpu_reference(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& ref_out) {
    std::vector<float> C(BLOCK_SIZE * BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += A[i * BLOCK_SIZE + k] * B[j * BLOCK_SIZE + k];
            }
            C[i * BLOCK_SIZE + j] = sum;
        }
    }
    ref_out.resize(BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        float row_max = -INFINITY;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            row_max = fmaxf(row_max, C[i * BLOCK_SIZE + j]);
        }
        ref_out[i] = row_max;
    }
}

#endif // UTILS_CUH
