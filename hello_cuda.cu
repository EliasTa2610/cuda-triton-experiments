#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(float* x, int n) { int i = blockIdx.x + threadIdx.x; if(i<n) x[i]=42.0f; }
int main() {
    int n=1<<20; size_t bytes=n*sizeof(float);
    float *d; cudaMalloc(&d, bytes);
    dim3 block(256), grid((n+block.x-1)/block.x);
    fill<<<grid,block>>>(d, n); cudaDeviceSynchronize();
    printf("launched %d blocks of %d threads\n", grid.x, block.x);
    cudaFree(d);
}
