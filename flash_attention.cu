#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>

#define BLOCK_SIZE 16

// FlashAttention kernel
// m[i] = running max for row i across processed K-tiles
// l[i] = running normalizer (sum of exp-shifted scores) for row i
__global__ void flash_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D) {
   
    assert(blockDim.x == D && blockDim.y <= N && "Block size must be (D, <=N)");
    assert(D % 32 == 0 && "D must be divisible by 32 for shuffle-based reduction");

    int row = threadIdx.y + blockIdx.x * blockDim.y; 
    int col = threadIdx.x;


    // Effective block size, accounting for last block which may have fewer rows
    int block_begin = blockIdx.x * blockDim.y;
    int block_size = min(blockDim.y, N - block_begin);
    
    extern __shared__ float K_block[];
    float* V_block = K_block + block_size * D;
    float* Q_block = V_block + block_size * D;
    float* O_block = Q_block + block_size * D;

    float* m_block = O_block + block_size * D;
    float* l_block = m_block + block_size;

    float* intermediate = l_block + block_size;

    // Initiaze per-row statistics
    if (col == 0 && row < N) {
        m_block[threadIdx.y] = -INFINITY;
        l_block[threadIdx.y] = 0.0f;
    }
    if (row < N) {
        Q_block[threadIdx.y * D + col] = Q[row * D + col]; 
        O_block[threadIdx.y * D + col] = 0.0f; // Initialize output block to zero 
    }

    //m[threadIdx.y + blockIdx.x * blockDim.y] = m_block[threadIdx.y];
    //l[threadIdx.y + blockIdx.x * blockDim.y] = l_block[threadIdx.y];
    __syncthreads();
    for (int j = 0 ; j < N ; j += block_size) {
        int kv_block_size = min(block_size, N - j); // Handle last block which may have fewer rows

        // Load K and V tiles
        int row_of_kv = threadIdx.y + j;
        if (row < N && row_of_kv < N) {
            K_block[threadIdx.y * D + col] = K[row_of_kv * D + col];
            V_block[threadIdx.y * D + col] = V[row_of_kv * D + col];
        }
        __syncthreads();

        if (row < N) {
            for (int c = col ; c < kv_block_size ; c += D) {
                float acc = 0.0f;
                for (int k = 0 ; k < D ; k++) {
                    acc += Q_block[threadIdx.y * D + k] * K_block[c * D + k];
                }
                intermediate[threadIdx.y * block_size + c] = acc;
            }
        }
        __syncthreads();
        
        float old_max; float old_normalizer;
        if (row < N) {
            old_max = m_block[threadIdx.y];
            old_normalizer = l_block[threadIdx.y];
        }
        __syncthreads();

        // Use leftmost warp to carry out online softmax 
        float thread_max = -INFINITY; float thread_normalizer = 0.0f;
        if (row < N && col < 32) {
            for (int k = col ; k < kv_block_size ; k += 32) {
                float new_thread_max = fmaxf(thread_max, intermediate[threadIdx.y * block_size + k]);
                thread_normalizer = thread_normalizer * expf(thread_max - new_thread_max) + expf(intermediate[threadIdx.y * block_size + k] - new_thread_max);
                thread_max = new_thread_max;
            }
        }
        __syncthreads();

        if (row < N && col < 32) {
            uint active_lanes = 0xffffffffu;
            for (int o = 16 ; o > 0 ; o /= 2) {
                float other_thread_max = __shfl_down_sync(active_lanes, thread_max, o);
                float other_thread_normalizer = __shfl_down_sync(active_lanes, thread_normalizer, o);

                float new_thread_max = fmaxf(thread_max, other_thread_max);
                thread_normalizer = thread_normalizer * expf(thread_max - new_thread_max) + other_thread_normalizer * expf(other_thread_max - new_thread_max);
                thread_max = new_thread_max;
            }
        }
        __syncthreads();

        if (col == 0 && row < N) {
            float new_max = fmaxf(old_max, thread_max);
            float new_normalizer = old_normalizer * expf(old_max - new_max) + thread_normalizer * expf(thread_max - new_max);
            l_block[threadIdx.y] = new_normalizer;
            m_block[threadIdx.y] = new_max;
        } 
        __syncthreads();

        // Accumulate in O block
        if (row < N) {
            // Undo normalization of current tile
            float reshifted_sum = O_block[threadIdx.y * D + col] * expf(old_max - m_block[threadIdx.y]);
            
            for (int k = 0; k < kv_block_size; k++) {
                reshifted_sum += expf(intermediate[threadIdx.y * block_size + k] - m_block[threadIdx.y]) * V_block[k * D + col];
            }
            O_block[threadIdx.y * D + col] = reshifted_sum; 
        }
        __syncthreads();
    }
    // Write to O
    if (row < N) {
        O[row * D + col] = O_block[threadIdx.y * D + col] / l_block[threadIdx.y]; // Final normalization
    }
}


static void print_matrix(const std::vector<float>& h, int rows, int cols, const char* title,
                         int max_rows = 3, int max_cols = 8) {
    std::cout << title << " (" << rows << "x" << cols << ")\n";
    int R = std::min(rows, max_rows);
    int C = std::min(cols, max_cols);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(5)
                      << h[r * cols + c] << ' ';
        }
        if (C < cols) std::cout << "...";
        std::cout << '\n';
    }
    if (R < rows) std::cout << "..." << std::endl;
}

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    // Problem size: single batch, single head for simplicity
    int N = 128;  // sequence length
    int D = 64;   // head dimension

    const size_t bytes_mat = static_cast<size_t>(N) * D * sizeof(float);

    // Host buffers
    std::vector<float> hQ(N * D), hK(N * D), hV(N * D), hO(N * D, 0.0f);

    // Seeded RNG for reproducible inputs
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * D; ++i) {
        hQ[i] = dist(rng);
        hK[i] = dist(rng);
        hV[i] = dist(rng);
    }

    // Device buffers
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaCheck(cudaMalloc(&dQ, bytes_mat), "cudaMalloc dQ");
    cudaCheck(cudaMalloc(&dK, bytes_mat), "cudaMalloc dK");
    cudaCheck(cudaMalloc(&dV, bytes_mat), "cudaMalloc dV");
    cudaCheck(cudaMalloc(&dO, bytes_mat), "cudaMalloc dO");

    // H2D copy
    cudaCheck(cudaMemcpy(dQ, hQ.data(), bytes_mat, cudaMemcpyHostToDevice), "Memcpy dQ");
    cudaCheck(cudaMemcpy(dK, hK.data(), bytes_mat, cudaMemcpyHostToDevice), "Memcpy dK");
    cudaCheck(cudaMemcpy(dV, hV.data(), bytes_mat, cudaMemcpyHostToDevice), "Memcpy dV");

    // Launch configuration
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    size_t maxPerBlock = prop.sharedMemPerBlock; 

    size_t smem_budget = (4 * BLOCK_SIZE * D + 2 * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);
    assert(smem_budget <= maxPerBlock && "Shared memory budget exceeds device limit!");

    dim3 block(D, BLOCK_SIZE);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks);

    flash_attention<<<grid, block, smem_budget>>>(dQ, dK, dV, dO, N, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    // D2H copy
    cudaCheck(cudaMemcpy(hO.data(), dO, bytes_mat, cudaMemcpyDeviceToHost), "Memcpy hO");

    // Print a small slice
    print_matrix(hO, N, D, "Output O");

    // Cleanup
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    std::cout << "Done (flash_attention)." << std::endl;
    return 0;
}
