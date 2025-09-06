#include "utils.cuh"
#include "kernels.cuh"

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    int kernel_choice = std::stoi(argv[1]);
    __nv_bfloat16 *d_A, *d_B; float* d_out;
    const int matrix_size = BLOCK_SIZE * BLOCK_SIZE;
    const int output_size = BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, output_size * sizeof(float)));
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);

    if (kernel_choice == 1) {
        Frag_standard_baseline<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    } else {
        Frag_swapped<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Nettoyage ---
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}
