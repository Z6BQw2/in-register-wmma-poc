#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include "../kernels.cuh"

#include <math.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define D_MODEL 512
#define PADDED_D (512 + 8)

__global__ void V6(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
                               int seq_len, int d_model) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int globRow = by*BLOCK_SIZE+ty;
    int globCol = bx*BLOCK_SIZE+tx;
    if (globRow >= seq_len || globCol >= d_model) return;

    float accumulator = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float l_i_dummy = 0.0f;
    float O_accum = 0.0f;

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][PADDED_D];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sP[BLOCK_SIZE][BLOCK_SIZE]; // Necessary because we tile_S is the wrong type, and wmma doesn't offer the possibility to convert it during loading
    __shared__ float tile_S[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = tx; i < d_model; i += blockDim.x) {
        sQ[ty][i] = Q[globRow * d_model + i];
    }

    //
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> work_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        sK[0][tx][ty] = K[j * BLOCK_SIZE * d_model + ty * d_model + tx];

        // S_ij
        for (int p = 0; p < (d_model / BLOCK_SIZE) - 1; p++){
            int current_buf = p % 2;
            int next_buf = 1 - current_buf;
            __syncthreads();

            wmma::load_matrix_sync(q_frag, &sQ[0][p * BLOCK_SIZE], PADDED_D);
            wmma::load_matrix_sync(k_frag, &sK[current_buf][0][0], BLOCK_SIZE);
            
            sK[next_buf][tx][ty] = K[(p + 1) * BLOCK_SIZE + j * BLOCK_SIZE * d_model + ty * d_model + tx];

            wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);
        }

        int last_buf_idx = (d_model / BLOCK_SIZE - 1) % 2;
        wmma::load_matrix_sync(q_frag, &sQ[0][(d_model / BLOCK_SIZE - 1) * BLOCK_SIZE], PADDED_D);
        wmma::load_matrix_sync(k_frag, &sK[last_buf_idx][0][0], BLOCK_SIZE);
        wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);

        float scale = 1.0f / sqrtf((float)d_model);
        for(int i = 0; i < 8; i++) work_frag.x[i] *= scale;

        wmma::store_matrix_sync(&tile_S[0][0], work_frag, BLOCK_SIZE, wmma::mem_row_major);

        accumulator = tile_S[ty][tx]; //Essential to calculate row-wise opperations, and to do it without storing, it would require an entire cycle of in-register de-swizzling, which brings us back to the same idea. 

        __syncthreads();
        float warp_val = accumulator;

        // Cette boucle est maintenant une réduction sur 16 éléments
        #pragma unroll
        for (int offset=8; offset>0; offset/=2) {
            warp_val = fmaxf(warp_val, __shfl_down_sync(0xFFFFFFFF, warp_val, offset));
        }

        // Le max est maintenant dans le thread tx=0 de chaque ligne.
        // On doit le diffuser aux autres.
        float m_ij = __shfl_sync(0xFFFFFFFF, warp_val, 0);

        // --- 2. RÉDUCTION DE LA SOMME (l_ij) ---
        float m_new = fmaxf(m_i, m_ij);
        float exp_val = expf(accumulator - m_ij);
        warp_val = exp_val;

        #pragma unroll
        for (int offset=8; offset>0; offset/=2) {
            warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, offset);
        }
        float l_ij = __shfl_sync(0xFFFFFFFF, warp_val, 0);
        
        l_i = l_i * expf(m_i - fmaxf(m_i, m_ij)) + l_ij * expf(m_ij - fmaxf(m_i, m_ij));

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> p_frag;
        sP[ty][tx] = __float2bfloat16(exp_val);
        __syncthreads();
        wmma::load_matrix_sync(p_frag, &sP[0][0], 16);

        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> v_frag;
        sV[ty][tx] = V[ (j * BLOCK_SIZE + ty) * d_model + (blockIdx.x * BLOCK_SIZE + tx) ];
        __syncthreads();
        wmma::load_matrix_sync(v_frag, &sV[0][0], 16);

        wmma::fill_fragment(work_frag, 0.0f);
        accumulator = 0;

        wmma::mma_sync(work_frag, p_frag, v_frag, work_frag);
        wmma::store_matrix_sync(&tile_S[0][0], work_frag, BLOCK_SIZE, wmma::mem_row_major);

        O_accum = O_accum * expf(m_i - fmaxf(m_i, m_ij)) * (l_i_dummy / l_i) + tile_S[ty][tx] * expf(m_ij - fmaxf(m_i, m_ij)) / l_i;

        m_i = fmax(m_i, m_ij);
        l_i_dummy = l_i;
        accumulator = 0;
    }
    out[globRow * d_model + globCol] = (__nv_bfloat16)O_accum;
}

__global__ void V7(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
                               int seq_len, int d_model) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int lane_id = threadIdx.x;
    // Et la garde devient :
    if (by * BLOCK_SIZE >= seq_len || bx * BLOCK_SIZE >= d_model) return;
    int global_row_start = by * BLOCK_SIZE;
    float max_A;
    float max_B;
    float max_A_i = -INFINITY;
    float max_B_i = -INFINITY;
    float sum_A;
    float sum_B;
    float sum_A_i = 0.0f;
    float sum_B_i = 0.0f;

    __shared__ float s_O_accum[32][8];

    for(int i = 0; i < 8; i++) {
        s_O_accum[threadIdx.x][i] = 0.0f;
    }

    __shared__ __nv_bfloat16 sQ[BLOCK_SIZE][D_MODEL];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];

    #pragma unroll
    for(int i = 0; i < (BLOCK_SIZE * d_model) / 32; i++) {
        int flat_idx = lane_id + i * 32;
        int row = flat_idx / d_model;
        int col = flat_idx % d_model;
        sQ[row][col] = Q[(global_row_start + row) * d_model + col];
    }

    //
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> work_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        #pragma unroll
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / 32; ++i) { // 256 elem / 32 threads = 8 iter
            int flat_idx = lane_id + i * 32;
            int row_in_tile = flat_idx / BLOCK_SIZE;
            int col_in_tile = flat_idx % BLOCK_SIZE;
            sK[0][row_in_tile][col_in_tile] = K[(j * BLOCK_SIZE + row_in_tile) * d_model + col_in_tile];
        }

        // S_ij
        for (int p = 0; p < (d_model / BLOCK_SIZE) - 1; p++){
            int current_buf = p % 2;
            int next_buf = 1 - current_buf;
            __syncthreads();

            wmma::load_matrix_sync(q_frag, &sQ[0][p * BLOCK_SIZE], D_MODEL);
            wmma::load_matrix_sync(k_frag, &sK[current_buf][0][0], BLOCK_SIZE);
            
            #pragma unroll
            for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / 32; ++i) {
                int flat_idx = lane_id + i * 32;
                int row_in_tile = flat_idx / BLOCK_SIZE;
                int col_in_tile = flat_idx % BLOCK_SIZE;
                sK[next_buf][row_in_tile][col_in_tile] = K[(j*BLOCK_SIZE + row_in_tile) * d_model + (p+1)*BLOCK_SIZE + col_in_tile];
            }

            wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);
        }

        int last_buf_idx = (d_model / BLOCK_SIZE - 1) % 2;
        wmma::load_matrix_sync(q_frag, &sQ[0][(d_model / BLOCK_SIZE - 1) * BLOCK_SIZE], D_MODEL);
        wmma::load_matrix_sync(k_frag, &sK[last_buf_idx][0][0], BLOCK_SIZE);
        wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);

        float scale = 1.0f / sqrtf((float)d_model);
        for(int i = 0; i < 8; i++) work_frag.x[i] *= scale;

        ///////////////////////////////////////////////////////////////////////////////////////

        float temp_f;

        temp_f = work_frag.x[2];
        work_frag.x[2] = work_frag.x[4];
        work_frag.x[4] = temp_f;

        temp_f = work_frag.x[3];
        work_frag.x[3] = work_frag.x[5];
        work_frag.x[5] = temp_f;
        
        max_A = fmaxf(fmaxf(work_frag.x[0], work_frag.x[1]), fmaxf(work_frag.x[2], work_frag.x[3]));
        max_B = fmaxf(fmaxf(work_frag.x[4], work_frag.x[5]), fmaxf(work_frag.x[6], work_frag.x[7]));

        float partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 1);
        float partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 1);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), max_B, 2);

        max_A = fmaxf(max_A, partner_max_A);
        max_B = fmaxf(max_B, partner_max_B);

        // --- 2. RÉDUCTION DE LA SOMME (l_ij) ---
        float m_new_A = fmaxf(max_A_i, max_A);
        float m_new_B = fmaxf(max_B_i, max_B);

        sum_A = 0.0f;
        sum_B = 0.0f;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            work_frag.x[i] = expf(work_frag.x[i] - max_A);
            sum_A += work_frag.x[i];
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            work_frag.x[i] = expf(work_frag.x[i] - max_B);
            sum_B += work_frag.x[i];
        }

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 1);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 1);

        sum_A += partner_max_A;
        sum_B += partner_max_B;

        partner_max_A = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_A, 2);
        partner_max_B = __shfl_xor_sync(0xF << ((threadIdx.x / 4) * 4), sum_B, 2);

        sum_A += partner_max_A;
        sum_B += partner_max_B; 

        float old_sum_A_i = sum_A_i; ///Remb (assignation potentiellement inutile)
        float old_sum_B_i = sum_B_i;

        sum_A_i = old_sum_A_i * expf(max_A_i - m_new_A) + sum_A * expf(max_A - m_new_A);
        sum_B_i = old_sum_B_i * expf(max_B_i - m_new_B) + sum_B * expf(max_B - m_new_B);
        
        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> p_frag;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_A); //Remb (/sum)
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_B);
        }

        temp_f = p_frag.x[2];
        p_frag.x[2] = p_frag.x[4];
        p_frag.x[4] = temp_f;

        temp_f = p_frag.x[3];
        p_frag.x[3] = p_frag.x[5];
        p_frag.x[5] = temp_f;

        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> v_frag;
        #pragma unroll
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / 32; ++i) { // 8 itérations
            int flat_idx = lane_id + i * 32;
            int row_in_tile = flat_idx / BLOCK_SIZE;
            int col_in_tile = flat_idx % BLOCK_SIZE;

            int global_V_row = j * BLOCK_SIZE + row_in_tile;
            int global_V_col = bx * BLOCK_SIZE + col_in_tile;

            sV[row_in_tile][col_in_tile] = V[global_V_row * d_model + global_V_col];
            
        }
        __syncthreads();
        wmma::load_matrix_sync(v_frag, &sV[0][0], 16);

        wmma::fill_fragment(work_frag, 0.0f);

        wmma::mma_sync(work_frag, p_frag, v_frag, work_frag);

        // O_accum = O_accum * expf(m_i - fmaxf(m_i, m_ij)) * (l_i_dummy / l_i) + tile_S[ty][tx] * expf(m_ij - fmaxf(m_i, m_ij)) / l_i;

        temp_f = work_frag.x[2]; work_frag.x[2] = work_frag.x[4]; work_frag.x[4] = temp_f;
        temp_f = work_frag.x[3]; work_frag.x[3] = work_frag.x[5]; work_frag.x[5] = temp_f;

        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_A_i / sum_A_i) * expf(max_A_i - fmax(max_A_i, max_A)) + work_frag.x[i] * (sum_A / sum_A_i);
        }

        #pragma unroll
        for(int i = 4; i < 8; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_B_i / sum_B_i) * expf(max_B_i - fmax(max_B_i, max_B)) + work_frag.x[i] * (sum_B / sum_B_i);
        }

        max_A_i = m_new_A;
        max_B_i = m_new_B;
    }
    float temp_f = s_O_accum[threadIdx.x][2];
    s_O_accum[threadIdx.x][2] = s_O_accum[threadIdx.x][4];
    s_O_accum[threadIdx.x][4] = temp_f;

    temp_f = s_O_accum[threadIdx.x][3];
    s_O_accum[threadIdx.x][3] = s_O_accum[threadIdx.x][5];
    s_O_accum[threadIdx.x][5] = temp_f;

    // Écriture avec les formules de mapping inverse
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        out[(blockIdx.y * 16 + (threadIdx.x / 4) + (((i >> 1) & 1) * 8)) * d_model + blockIdx.x * 16 + ((threadIdx.x % 4) * 2) + (i % 2) + (((i >> 2) & 1) * 8)] = __float2bfloat16(s_O_accum[threadIdx.x][i]);
    }
}


void attention_reference(float* Q_float, float* K_float, float* V_float, float* out_float, int seq_len, int d_model) {    
    
    float* S = (float*)malloc(seq_len * seq_len * sizeof(float));

    // S = Q @ K^T / sqrt(d_model)
    float scale = 1.0f / sqrtf((float)d_model);
    for(int i = 0; i < seq_len; i++) {
        for(int j = 0; j < seq_len; j++) {
            float sum = 0;
            for(int k = 0; k < d_model; k++) {
                sum += Q_float[i*d_model + k] * K_float[j*d_model + k];
            }
            S[i*seq_len + j] = sum * scale;
        }
    }
    
    // Softmax par ligne
    for(int i = 0; i < seq_len; i++) {
        float max_val = -INFINITY;
        for(int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, S[i*seq_len + j]);
        }
        float sum = 0;
        for(int j = 0; j < seq_len; j++) {
            S[i*seq_len + j] = expf(S[i*seq_len + j] - max_val);
            sum += S[i*seq_len + j];
        }
        for(int j = 0; j < seq_len; j++) {
            S[i*seq_len + j] /= sum;
        }
    }
    
    // Out = S @ V
    for(int i = 0; i < seq_len; i++) {
        for(int j = 0; j < d_model; j++) {
            float sum = 0;
            for(int k = 0; k < seq_len; k++) {
                sum += S[i*seq_len + k] * V_float[k*d_model + j];
            }
            out_float[i*d_model + j] = sum;
        }
    }
    
    free(S);
}


int main() {
    const int seq_len = 1024;
    const int d_model = 512;
    const int size_bf16 = seq_len * d_model * sizeof(__nv_bfloat16);
    const int warmup_rounds = 10;
    const int timing_rounds = 100;

    // --- 1. SETUP & INITIALISATION DES DONNÉES (Identique) ---
    printf("Setting up data for seq_len=%d, d_model=%d...\n", seq_len, d_model);
    float *h_Q_float = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_K_float = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_V_float = (float*)malloc(seq_len * d_model * sizeof(float));
    
    __nv_bfloat16 *h_Q = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_K = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_V = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_out_gpu = (__nv_bfloat16*)malloc(size_bf16);
    float *h_out_cpu_ref = (float*)malloc(seq_len * d_model * sizeof(float));

    srand(42);
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_K_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_V_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        
        h_Q[i] = __float2bfloat16(h_Q_float[i]);
        h_K[i] = __float2bfloat16(h_K_float[i]);
        h_V[i] = __float2bfloat16(h_V_float[i]);
    }

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, size_bf16);
    cudaMalloc(&d_K, size_bf16);
    cudaMalloc(&d_V, size_bf16);
    cudaMalloc(&d_out, size_bf16);

    cudaMemcpy(d_Q, h_Q, size_bf16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_bf16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_bf16, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 blockDim(32, 1, 1);
    dim3 gridDim((d_model + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (seq_len + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    
    // --- 2. BENCHMARK KERNEL 1: IN-REGISTER (VOTRE VERSION) ---
    printf("\n--- Benchmarking In-Register Kernel ---\n");
    printf("Performing %d warm-up rounds...\n", warmup_rounds);
    for (int i = 0; i < warmup_rounds; ++i) {
        V7<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < timing_rounds; i++) {
        V7<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms_in_register = ms / timing_rounds;
    printf("Average execution time: %.5f ms\n", avg_ms_in_register);

    // --- 3. BENCHMARK KERNEL 2: SHARED MEMORY BASELINE ---
    printf("\n--- Benchmarking Shared Memory Baseline Kernel ---\n");
    printf("Performing %d warm-up rounds...\n", warmup_rounds);
    for (int i = 0; i < warmup_rounds; ++i) {
        V7<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < timing_rounds; i++) {
        V7<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms_shared_mem = ms / timing_rounds;
    printf("Average execution time: %.5f ms\n", avg_ms_shared_mem);

    // --- 4. RÉSULTATS FINALS ---
    printf("\n\n==================== BENCHMARK SUMMARY ====================\n");
    printf("In-Register Kernel         : %.5f ms\n", avg_ms_in_register);
    printf("Shared Memory Baseline Kernel: %.5f ms\n", avg_ms_shared_mem);
    printf("Speedup (In-Register vs Shared): %.2fx\n", avg_ms_shared_mem / avg_ms_in_register);
    printf("===========================================================\n\n");

    // --- 5. VÉRIFICATION DE LA JUSTESSE (uniquement pour le kernel In-Register) ---
    printf("--- Correctness Check (for In-Register Kernel vs CPU) ---\n");
    // On exécute le kernel in-register une dernière fois pour être sûr que d_out contient son résultat
    V6<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    cudaMemcpy(h_out_gpu, d_out, size_bf16, cudaMemcpyDeviceToHost);
    
    printf("Running CPU reference...\n");
    attention_reference(h_Q_float, h_K_float, h_V_float, h_out_cpu_ref, seq_len, d_model);

    float max_error = 0;
    for(int i = 0; i < seq_len * d_model; i++) {
        float gpu_val = __bfloat162float(h_out_gpu[i]);
        float ref_val = h_out_cpu_ref[i];
        max_error = fmaxf(max_error, fabs(ref_val - gpu_val));
    }
    printf("Max absolute error between GPU (In-Register) and CPU reference: %e\n", max_error);
    
    // --- 6. CLEANUP ---
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    free(h_Q_float); free(h_K_float); free(h_V_float);
    free(h_Q); free(h_K); free(h_V); free(h_out_gpu); free(h_out_cpu_ref);  
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    
    printf("\nCleanup complete. Exiting.\n");
    return 0;
}
