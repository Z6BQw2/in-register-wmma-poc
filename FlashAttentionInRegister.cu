#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define PADDED_D (512 + 8)

__global__ void flash_attention_kernel(__nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V, __nv_bfloat16* out, 
                               int seq_len, int d_model) {
    // int bx = blockIdx.x;
    // int by = blockIdx.y;
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // int globRow = by*BLOCK_SIZE+ty;
    // int globCol = bx*BLOCK_SIZE+tx;

    if (blockIdx.x*BLOCK_SIZE >= d_model || blockIdx.y*BLOCK_SIZE >= seq_len) return;
    unsigned int* vals;
    float max_A;
    float max_B;
    float max_A_i = -INFINITY;
    float max_B_i = -INFINITY;
    float sum_A;
    float sum_B;
    float sum_A_i = 0.0f;
    float sum_B_i = 0.0f;

    __shared__ __nv_bfloat16 sQ_full[BLOCK_SIZE][PADDED_D];
    __shared__ __nv_bfloat16 sK[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __nv_bfloat16 sV[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_O_accum[32][8];

    for(int i = 0; i < 256; i++) {
        int flat_idx = threadIdx.x + i * 32;
        int row = flat_idx / 512;
        int col = flat_idx % 512;
        if(row < 16) {
            sQ_full[row][col] = Q[(blockIdx.y * 16 + row) * d_model + col];
        }
    }

    for(int i = 0; i < 8; i++) {
        s_O_accum[threadIdx.x][i] = 0.0f;
    }

    // BOUCLE FLASH (externe) - Itère sur les blocs de K/V
    for (int j = 0; j < seq_len / BLOCK_SIZE; ++j) {

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> work_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        // A. PROLOGUE: Remplir le premier buffer (sK[0]).
        for(int i = 0; i < 8; i++) {
            // ... (logique de calcul d'index inchangée)
            int tile_idx = threadIdx.x + i * 32;
            int row_in_tile = tile_idx / BLOCK_SIZE;
            int col_in_tile = tile_idx % BLOCK_SIZE;
            int global_K_row = j * BLOCK_SIZE + row_in_tile;
            int global_K_col = 0 + col_in_tile;
            sK[0][row_in_tile][col_in_tile] = K[global_K_row * d_model + global_K_col];
        }

        // B. BOUCLE PRINCIPALE: Le pipeline.
        for (int p = 0; p < d_model / BLOCK_SIZE - 1; ++p) {
            __syncthreads();
            // Choisit quel buffer est le 'current' et quel est le 'next' pour CETTE itération
            int current_buf_idx = p % 2;
            int next_buf_idx = 1 - current_buf_idx;

            // 1. Pré-charger la PROCHAINE tuile K (p+1) dans le buffer "next".
            for(int i = 0; i < 8; i++) {
                // ... (logique de calcul d'index inchangée)
                int tile_idx = threadIdx.x + i * 32;
                int row_in_tile = tile_idx / BLOCK_SIZE;
                int col_in_tile = tile_idx % BLOCK_SIZE;
                int global_K_row = j * BLOCK_SIZE + row_in_tile;
                int global_K_col = (p + 1) * BLOCK_SIZE + col_in_tile;
                sK[next_buf_idx][row_in_tile][col_in_tile] = K[global_K_row * d_model + global_K_col];
            }

            // 2. Calculer avec la tuile ACTUELLE (p), qui est déjà dans le buffer "current".
            wmma::load_matrix_sync(q_frag, &sQ_full[0][p * BLOCK_SIZE], PADDED_D);
            wmma::load_matrix_sync(k_frag, &sK[current_buf_idx][0][0], BLOCK_SIZE);
            wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);

        }

        // C. EPILOGUE: Calculer la TOUTE DERNIERE tuile.
        // On doit déterminer dans quel buffer elle a été chargée en dernier.
        int last_buf_idx = (d_model / BLOCK_SIZE - 1) % 2;
        wmma::load_matrix_sync(q_frag, &sQ_full[0][(d_model / BLOCK_SIZE - 1) * BLOCK_SIZE], PADDED_D);
        wmma::load_matrix_sync(k_frag, &sK[last_buf_idx][0][0], BLOCK_SIZE);
        wmma::mma_sync(work_frag, q_frag, k_frag, work_frag);
        // ----- FIN DE LA SECTION QK^T -----

        // Le reste du code (softmax, etc.) commence ici.
        for(int i = 0; i < 8; i++) {
            work_frag.x[i] *= 1.0f / sqrtf((float)d_model);
        }

        float temp_f;

        // Échange le bloc {x[2], x[3]} avec le bloc {x[4], x[5]}
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

        sum_A = 0.0f;
        sum_B = 0.0f;
        for(int i = 0; i < 4; ++i) {
            work_frag.x[i] = expf(work_frag.x[i] - max_A);
            sum_A += work_frag.x[i];
        }
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

        float old_sum_A_i = sum_A_i;
        float old_sum_B_i = sum_B_i;

        sum_A_i = old_sum_A_i * expf(max_A_i - fmax(max_A_i, max_A)) + sum_A * expf(max_A - fmax(max_A_i, max_A));
        sum_B_i = old_sum_B_i * expf(max_B_i - fmax(max_B_i, max_B)) + sum_B * expf(max_B - fmax(max_B_i, max_B));

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> p_frag;

        for(int i = 0; i < 4; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_A);
        }
        for(int i = 4; i < 8; ++i) {
            p_frag.x[i] = __float2bfloat16(work_frag.x[i] / sum_B);
        }

        vals = reinterpret_cast<unsigned int*>(p_frag.x); //Un swap pour 2 valeurs: La reinterprétation prend 2 bits soit deux bfloat16, ce qui permet un swap en une seule instruction. Y'a pas de petites économies :)
        unsigned int temp = vals[1];
        vals[1] = vals[2];
        vals[2] = temp;

        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __nv_bfloat16, wmma::row_major> v_frag;
        wmma::fill_fragment(work_frag, 0.0f);

        for(int i = 0; i < 8; i++) {
            int tile_idx = threadIdx.x + i * 32;
            int row_in_tile = tile_idx / BLOCK_SIZE;
            int col_in_tile = tile_idx % BLOCK_SIZE;

            int global_V_row = j * BLOCK_SIZE + row_in_tile;

            int global_V_col = blockIdx.x * BLOCK_SIZE + col_in_tile;

            if (global_V_row < seq_len && global_V_col < d_model) {
                sV[row_in_tile][col_in_tile] = V[global_V_row * d_model + global_V_col];
            } else {
                sV[row_in_tile][col_in_tile] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(v_frag, &sV[0][0], 16);
        wmma::mma_sync(work_frag, p_frag, v_frag, work_frag);

        temp_f = work_frag.x[2];
        work_frag.x[2] = work_frag.x[4];
        work_frag.x[4] = temp_f;

        temp_f = work_frag.x[3];
        work_frag.x[3] = work_frag.x[5];
        work_frag.x[5] = temp_f;

        for(int i = 0; i < 4; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_A_i / sum_A_i) * expf(max_A_i - fmax(max_A_i, max_A)) + work_frag.x[i] * (sum_A / sum_A_i);
        }

        for(int i = 4; i < 8; ++i) {
            s_O_accum[threadIdx.x][i] = s_O_accum[threadIdx.x][i] * (old_sum_B_i / sum_B_i) * expf(max_B_i - fmax(max_B_i, max_B)) + work_frag.x[i] * (sum_B / sum_B_i);
        }

        max_A_i = fmax(max_A_i, max_A);
        max_B_i = fmax(max_B_i, max_B);

        __syncthreads();
    }

    float temp_f = s_O_accum[threadIdx.x][2];
    s_O_accum[threadIdx.x][2] = s_O_accum[threadIdx.x][4];
    s_O_accum[threadIdx.x][4] = temp_f;

    temp_f = s_O_accum[threadIdx.x][3];
    s_O_accum[threadIdx.x][3] = s_O_accum[threadIdx.x][5];
    s_O_accum[threadIdx.x][5] = temp_f;

    // Écriture avec les formules de mapping inverse
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
    const int warmup_rounds = 20;
    const int run = 10;
    cudaError_t err;

    // Host arrays en float pour l'initialisation
    float *h_Q_float = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_K_float = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_V_float = (float*)malloc(seq_len * d_model * sizeof(float));
    
    // Host arrays en bfloat16
    __nv_bfloat16 *h_Q = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_K = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_V = (__nv_bfloat16*)malloc(size_bf16);
    __nv_bfloat16 *h_out_1 = (__nv_bfloat16*)malloc(size_bf16);
    float *h_out_2 = (float*)malloc(seq_len * d_model * sizeof(float));

    srand(42);
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_K_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        h_V_float[i] = ((float)rand()/RAND_MAX) - 0.5f;
        
        // Conversion float -> bfloat16
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

    dim3 blockDim(32, 1, 1);  // Un seul warp
    dim3 gridDim((d_model + 15) / 16, (seq_len + 15) / 16, 1);

    // --- WARM-UP ROUNDS ---
    printf("Performing %d warm-up rounds...\n", warmup_rounds);
    for (int i = 0; i < warmup_rounds; ++i) {
        flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < run; ++i) {
        flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }

    cudaMemcpy(h_out_1, d_out, size_bf16, cudaMemcpyDeviceToHost);

    attention_reference(h_Q_float, h_K_float, h_V_float, h_out_2, seq_len, d_model);

    printf("\nReference (CPU):\n");
    for(int i=0; i<10; i++){
        printf("%8.5f ", h_out_2[i]);
    }
    printf("\n\nGPU Kernel:\n");
    for(int i=0; i<10; i++){
        printf("%8.5f ", __bfloat162float(h_out_1[i]));
    }

    // Calcul de l'erreur sur TOUTE la matrice
    float max_error = 0;
    float sum_error = 0;
    float sum_squared_error = 0;
    int error_count = 0;

    for(int i = 0; i < seq_len * d_model; i++) {
        float gpu_val = __bfloat162float(h_out_1[i]);
        float ref_val = h_out_2[i];
        float diff = fabs(ref_val - gpu_val);
        
        max_error = fmaxf(max_error, diff);
        sum_error += diff;
        sum_squared_error += diff * diff;
        
        // Compte les erreurs > seuil
        if(diff > 1e-3) error_count++;
    }

    float mean_error = sum_error / (seq_len * d_model);
    float rms_error = sqrtf(sum_squared_error / (seq_len * d_model));

    printf("\n=== Error Analysis ===\n");
    printf("Max absolute error: %e\n", max_error);
    printf("Mean absolute error: %e\n", mean_error);
    printf("RMS error: %e\n", rms_error);
    printf("Values with error > 1e-3: %d / %d (%.2f%%)\n", 
        error_count, seq_len * d_model, 
        100.0f * error_count / (seq_len * d_model));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < 100; i++) {
        flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_out, seq_len, d_model);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms/100);

    free(h_Q_float); free(h_K_float); free(h_V_float);
    free(h_Q); free(h_K); free(h_V); free(h_out_1); free(h_out_2);  
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);

    return 0;
}