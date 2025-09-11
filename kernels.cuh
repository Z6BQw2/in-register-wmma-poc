#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16

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

    int lane = threadIdx.x;
    int my_row = lane % 16;
    int start_col = (lane < 16) ? 0 : 8;
    
    float local_max = -INFINITY;
    for (int i = 0; i < 8; i++) {
        local_max = fmaxf(local_max, sD[my_row][start_col + i]);
    }

    unsigned mask = (1u << lane) | (1u << (lane ^ 16));
    float partner_max = __shfl_xor_sync(mask, local_max, 16);
    float final_row_max = fmaxf(local_max, partner_max);

    if (lane < 16) {
        out[my_row] = final_row_max;
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
        out[row_group] = max_A;
        out[row_group + 8] = max_B;
    }
}

#endif // KERNELS_CUH
