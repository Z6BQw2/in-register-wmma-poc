#include "utils.cuh"
#include "kernels.cuh"
#include <random>
#include <algorithm>

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
