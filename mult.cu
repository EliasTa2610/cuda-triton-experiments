// Tiled matrix multiplication (shared-memory tiles)
// - BLOCK_WIDTH x BLOCK_WIDTH tiles of A and B are staged in __shared__
// - Accumulate C tile via inner k loop over the tile width
// - Grid covers ceil(n / BLOCK_WIDTH) tiles in each dimension
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 16

__global__ void matmul(const float* A, const float* B, float* C, int n) { 
    __shared__ float A_tile[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float B_tile[BLOCK_WIDTH][BLOCK_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float value = 0.0f;
    for (int t = 0 ; t < (n + BLOCK_WIDTH - 1) / BLOCK_WIDTH ; t++) {
        if (row < n && col < n) {
            A_tile[ty][tx] = A[row*n + (t*BLOCK_WIDTH + tx)];
            B_tile[ty][tx] = B[(t*BLOCK_WIDTH + ty)*n + col];
        }
        else {
            A_tile[ty][tx] = 0.0f;
            B_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0 ; k < BLOCK_WIDTH ; k++)
            value += A_tile[ty][k] * B_tile[k][tx];
        __syncthreads();
    }
    if (row < n && col < n) {
        C[row*n + col] = value;
    }
}

int main() {
    int n=1000; 
    size_t bytes=n*n*sizeof(float);
    float *A; float *B; float *C;

    std::vector<float> hA(n*n, 1.0f);    
    std::vector<float> hB(n*n, 2.0f);
    std::vector<float> hC(n*n, -1.0f);

    cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);

    cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid((n+BLOCK_WIDTH-1)/BLOCK_WIDTH, (n+BLOCK_WIDTH-1)/BLOCK_WIDTH);
    matmul<<<grid,block>>>(A, B, C, n); 
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost);
   
    printf("C[0,0] = %f\n", hC[0]);

    cudaFree(A); cudaFree(B); cudaFree(C);
}
