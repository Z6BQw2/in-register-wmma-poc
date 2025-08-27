#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

using namespace nvcuda;

__global__ void Frag_standard_baseline(const __nv_bfloat16* A, const __nv_bfloat16* B, float* out) {
    __shared__ float sD[BLOCK_SIZE][BLOCK_SIZE];

    wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> acc_frag;
    
    wmma::load_matrix_sync(a_frag, A, BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag, B, BLOCK_SIZE);
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(&sD[0][0], acc_frag, BLOCK_SIZE, wmma::mem_row_major);
    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE) {
        int my_row = threadIdx.x;
        float row_max = -INFINITY;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            row_max = fmaxf(row_max, sD[my_row][i]);
        }
        out[my_row] = row_max;
    }
}

__global__ void Frag_swapped(const __nv_bfloat16* A, const __nv_bfloat16* B, float* out) {
    wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> acc_frag;

    wmma::load_matrix_sync(a_frag, A, BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag, B, BLOCK_SIZE);
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    float temp_f = acc_frag.x[2];
    acc_frag.x[2] = acc_frag.x[4];
    acc_frag.x[4] = temp_f;
    temp_f = acc_frag.x[3];
    acc_frag.x[3] = acc_frag.x[5];
    acc_frag.x[5] = temp_f;

    float max_A = fmaxf(fmaxf(acc_frag.x[0], acc_frag.x[1]), fmaxf(acc_frag.x[2], acc_frag.x[3]));
    float max_B = fmaxf(fmaxf(acc_frag.x[4], acc_frag.x[5]), fmaxf(acc_frag.x[6], acc_frag.x[7]));

    unsigned int mask = 0xF << ((threadIdx.x / 4) * 4);
    max_A = fmaxf(max_A, __shfl_xor_sync(mask, max_A, 1));
    max_A = fmaxf(max_A, __shfl_xor_sync(mask, max_A, 2));
    max_B = fmaxf(max_B, __shfl_xor_sync(mask, max_B, 1));
    max_B = fmaxf(max_B, __shfl_xor_sync(mask, max_B, 2));

    if(threadIdx.x % 4 == 0){
        int row_group = threadIdx.x / 4;
        
        int row_idx_A = row_group;
        int row_idx_B = row_group + 8;

        out[row_idx_A] = max_A;
        out[row_idx_B] = max_B;
    }
}

// Ref CPU
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

int main() {
    int deviceId;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    std::cout << "Utilisation du GPU: " << props.name << std::endl;

    const int matrix_size = BLOCK_SIZE * BLOCK_SIZE;
    std::vector<float> h_A(matrix_size);
    std::vector<float> h_B(matrix_size);
    std::vector<__nv_bfloat16> h_A_bf16(matrix_size);
    std::vector<__nv_bfloat16> h_B_bf16(matrix_size);

    std::mt19937 gen(1337); // Seed pour la reproductibilité
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int i = 0; i < matrix_size; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
        h_A_bf16[i] = __float2bfloat16(h_A[i]);
        h_B_bf16[i] = __float2bfloat16(h_B[i]);
    }
    
    __nv_bfloat16 *d_A, *d_B;
    float* d_out;
    const int output_size = BLOCK_SIZE;
    std::vector<float> h_out(output_size);

    CUDA_CHECK(cudaMalloc(&d_A, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_bf16.data(), matrix_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_bf16.data(), matrix_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Ref CPU
    std::vector<float> ref_out;
    cpu_reference(h_A, h_B, ref_out);

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // Benchmark Baseline
    std::cout << "\n--- Lancement Kernel 1 (Baseline) ---" << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    Frag_standard_baseline<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Temps: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validation
    int errors = 0;
    float epsilon = 1e-2;
    for(int i = 0; i < output_size; ++i) {
        if (std::abs(h_out[i] - ref_out[i]) > epsilon) errors++;
    }
    std::cout << "Validation: " << (errors == 0 ? "PASS" : "FAIL") << " (" << errors << " erreurs)" << std::endl;

    // Benchmark In-Reg
    std::cout << "\n--- Lancement Kernel 2 (In-Register) ---" << std::endl;
    CUDA_CHECK(cudaMemset(d_out, 0, output_size * sizeof(float))); 
    CUDA_CHECK(cudaEventRecord(start));
    Frag_swapped<<<gridDim, blockDim>>>(d_A, d_B, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Temps: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    errors = 0;
    for(int i = 0; i < output_size; ++i) {
        if (std::abs(h_out[i] - ref_out[i]) > epsilon) errors++;
    }
    std::cout << "Validation: " << (errors == 0 ? "PASS" : "FAIL") << " (" << errors << " erreurs)" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}