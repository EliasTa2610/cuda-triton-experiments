// naive_softmax.cu

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Per-row softmax
// Each block handles one row; threads iterate columns.
// 1) Find per-row max (reduce across columns);
// 2) Subtract max and exponentiate
// 3) Sum exps (reduce across columns);
// 4) Divide by sum to normalize
// Monoid structure:
// (m_1, s_1) ⊕ (m_2, s_2) = (m, s_1 exp(m_1 - m) + s_2 exp(m_2 - m)) where m = max(m_1, m_2),
// where s is max-shifted sum of exp.
__global__ void online_softmax(const float* __restrict__ logits,
                              float* __restrict__ probs,
                              int rows,
                              int cols) {

    // Phase 0: Make block size effectively a power of two
    int leftmost_power_of_two = 1u << (31 - __clz(static_cast<unsigned>(blockDim.x)));
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= rows) return;

    // Phase 1: Consolidate right slack to effective block 
    extern __shared__ float shared_max[];
    float* shared_exp_sum = shared_max + leftmost_power_of_two;

    if (col < leftmost_power_of_two) {
        float pmax = -INFINITY;
        float pexp = 0.0f;
        for (int tx = col; tx < cols; tx += leftmost_power_of_two) {
            float new_pmax = fmaxf(pmax, logits[row * cols + tx]);
            pexp = pexp * expf(pmax - new_pmax) + expf(logits[row * cols + tx] - new_pmax);
            pmax = new_pmax;
        }
        shared_max[col] = pmax;
        shared_exp_sum[col] = pexp;
    }
    __syncthreads();

    // Phase 2: Fold block
    for (int stride = leftmost_power_of_two / 2; stride >= 1; stride /= 2) {
        if (col < stride) {
            float m1 = shared_max[col];
            float m2 = shared_max[col + stride];
            float m = fmaxf(m1, m2);
            shared_max[col] = m;
            shared_exp_sum[col] =
                shared_exp_sum[col] * expf(m1 - m) + shared_exp_sum[col + stride] * expf(m2 - m);
        }
        __syncthreads();
    }

    // Phase 3
    float global_max = shared_max[0];
    float global_exp_sum = shared_exp_sum[0];

    if (col < leftmost_power_of_two) {
        for (int tx = col; tx < cols; tx += leftmost_power_of_two) {
            float prob = expf(logits[row * cols + tx] - global_max) / global_exp_sum;
            probs[row * cols + tx] = prob;
        }
    }
}

static void print_matrix(const std::vector<float>& h, int rows, int cols, const char* title) {
    std::cout << title << " (" << rows << "x" << cols << ")\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(5) << h[r * cols + c]
                      << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Problem size (rows: batch, cols: features per row);
    int rows = 2;
    int cols = 4;
    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        if (rows <= 0 || cols <= 0) {
            std::cerr << "Rows and cols must be positive." << std::endl;
            return EXIT_FAILURE;
        }
    }

    const int n = rows * cols;
    std::vector<float> h_logits(n);
    std::vector<float> h_probs(n, 0.0f);

    // Seeded RNG for reproducible inputs
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < n; ++i) {
        h_logits[i] = dist(rng);
    }

    print_matrix(h_logits, rows, cols, "Input logits");

    // Device allocations
    float* d_logits = nullptr;
    float* d_probs = nullptr;
    cudaMalloc(&d_logits, n * sizeof(float));
    cudaMalloc(&d_probs, n * sizeof(float));

    // Initialize output to zero so we have deterministic values until kernel is implemented
    cudaMemset(d_probs, 0, n * sizeof(float));

    // H2D copy
    cudaMemcpy(d_logits, h_logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration: one block per row, threads over columns
    int threads = cols;
    if (threads > 1024) threads = 1024;  // hardware limit; for large cols, implement striding inside kernel
    dim3 block(threads);
    dim3 grid(rows);

    size_t shmem_bytes = 2 * static_cast<size_t>(threads) * sizeof(float);

    online_softmax<<<grid, block, shmem_bytes>>>(d_logits, d_probs, rows, cols);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // D2H copy
    cudaMemcpy(h_probs.data(), d_probs, n * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(h_probs, rows, cols,
                 "Output probs (TODO: will be zeros until kernel is implemented)");

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_probs);

    std::cout << "Done. Implement the kernel in softmax_naive() to compute true softmax."
              << std::endl;
    return EXIT_SUCCESS;
}
