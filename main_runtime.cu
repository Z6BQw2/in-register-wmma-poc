#include "utils.cuh"
#include "kernels.cuh"
#include <iomanip>

int main() {
    // --- Initialisation (identique à avant) ---
    __nv_bfloat16 *d_A, *d_B; float* d_out;
    const int matrix_size = BLOCK_SIZE * BLOCK_SIZE;
    const int output_size = BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, output_size * sizeof(float)));
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // --- Mesure Kernel 1 ---
    CUDA_CHECK(cudaEventRecord(start));
    Frag_standard_baseline<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float latency_baseline = ms;

    // --- Mesure Kernel 2 ---
    CUDA_CHECK(cudaEventRecord(start));
    Frag_swapped<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float latency_in_register = ms;

    // --- Impression du résultat pour le script bash ---
    std::cout << std::fixed << std::setprecision(8) << latency_baseline << " " << latency_in_register << std::endl;

    // --- Nettoyage ---
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
