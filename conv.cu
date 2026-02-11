#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>  // ADD THIS

#define FILTER_RADIUS 2                                     // e.g., 5x5 kernel
#define OUTER_TILE_DIM 32                                    // tile loaded into smem
#define INNER_TILE_DIM (OUTER_TILE_DIM - 2*FILTER_RADIUS)       // valid output per block

// filter in constant memory (flattened (2R+1)x(2R+1))
__constant__ float F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

// Variant 1: Block sized to OUTER tile (OUTER_TILE_DIM x OUTER_TILE_DIM)
// - Each thread loads exactly one element of the OUTER tile (incl. halo) into shared memory
// - Only INNER region threads compute and write outputs
__global__ void convolution_var_1(
    const float* __restrict__ N, float* __restrict__ P,
    int width, int height)
{
    // global coords for the element this thread will load into smem (with halo)
    int col = blockIdx.x * INNER_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * INNER_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[OUTER_TILE_DIM][OUTER_TILE_DIM];

    // load input tile with zero-padding outside image
    if (row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // threads that own a *valid output* inside the tile (skip halo lanes)
    int tCol = threadIdx.x - FILTER_RADIUS;
    int tRow = threadIdx.y - FILTER_RADIUS;

    if (tCol >= 0 && tCol < INNER_TILE_DIM && tRow >= 0 && tRow < INNER_TILE_DIM) {
        int out_col = blockIdx.x * INNER_TILE_DIM + tCol;
        int out_row = blockIdx.y * INNER_TILE_DIM + tRow;

        if (out_row < height && out_col < width) {
            float acc = 0.0f;
            #pragma unroll
            for (int fr = -FILTER_RADIUS; fr <= FILTER_RADIUS; ++fr) {
                #pragma unroll
                for (int fc = -FILTER_RADIUS; fc <= FILTER_RADIUS; ++fc) {
                    float w = F[(fr + FILTER_RADIUS)*(2*FILTER_RADIUS+1) + (fc + FILTER_RADIUS)];
                    acc += w * N_s[threadIdx.y + fr][threadIdx.x + fc];
                }
            }
            P[out_row * width + out_col] = acc;
        }
    }
}

// Variant 2: Inner threads compute (prefer INNER_TILE_DIM x INNER_TILE_DIM)
// - Inner threads load the interior; halo populated via edge/corner conditionals
// - Fewer idle threads than var_1; more control flow during load
__global__ void convolution_var_2(
    const float* __restrict__ N, float* __restrict__ P,
    int width, int height)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx_local = threadIdx.x - FILTER_RADIUS; // maps -FILTER_RADIUS..INNER_TILE_DIM+FILTER_RADIUS-1
    int ty_local = threadIdx.y - FILTER_RADIUS;

    int row = by * INNER_TILE_DIM + ty_local;
    int col = bx * INNER_TILE_DIM + tx_local;

    __shared__ float N_s[OUTER_TILE_DIM][OUTER_TILE_DIM];

    // Inner tile: threads whose local indices fall in [0, INNER_TILE_DIM)
    if (tx_local >= 0 && tx_local < INNER_TILE_DIM && ty_local >= 0 && ty_local < INNER_TILE_DIM) {
        if (row >= 0 && row < height && col >= 0 && col < width)
            N_s[ty_local + FILTER_RADIUS][tx_local + FILTER_RADIUS] = N[row * width + col];
        else
            N_s[ty_local + FILTER_RADIUS][tx_local + FILTER_RADIUS] = 0.0f;
    }

    // Edges: top and bottom rows (inner columns)
    if (ty_local == 0 && tx_local >= 0 && tx_local < INNER_TILE_DIM && col < width) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            if (row >= i)
                N_s[ty_local - i + FILTER_RADIUS][tx_local + FILTER_RADIUS] = N[(row - i) * width + col];
            else
                N_s[ty_local - i + FILTER_RADIUS][tx_local + FILTER_RADIUS] = 0.0f;
        }
    }
    if (ty_local == INNER_TILE_DIM - 1 && tx_local >= 0 && tx_local < INNER_TILE_DIM && col < width) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            if (row + i < height)
                N_s[ty_local + i + FILTER_RADIUS][tx_local + FILTER_RADIUS] = N[(row + i) * width + col];
            else
                N_s[ty_local + i + FILTER_RADIUS][tx_local + FILTER_RADIUS] = 0.0f;
        }
    }

    // Edges: left and right columns (inner rows)
    if (tx_local == 0 && ty_local >= 0 && ty_local < INNER_TILE_DIM && row < height) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            if (col >= i)
                N_s[ty_local + FILTER_RADIUS][tx_local - i + FILTER_RADIUS] = N[row * width + col - i];
            else
                N_s[ty_local + FILTER_RADIUS][tx_local - i + FILTER_RADIUS] = 0.0f;
        }
    }
    if (tx_local == INNER_TILE_DIM - 1 && ty_local >= 0 && ty_local < INNER_TILE_DIM && row < height) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            if (col + i < width)
                N_s[ty_local + FILTER_RADIUS][tx_local + i + FILTER_RADIUS] = N[row * width + col + i];
            else
                N_s[ty_local + FILTER_RADIUS][tx_local + i + FILTER_RADIUS] = 0.0f;
        }
    }

    // Corners
    // top-left
    if (tx_local == 0 && ty_local == 0) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            for (int j = 1; j <= FILTER_RADIUS; ++j) {
                if (row - i >= 0 && col - j >= 0)
                    N_s[ty_local - i + FILTER_RADIUS][tx_local - j + FILTER_RADIUS] = N[(row - i) * width + col - j];
                else
                    N_s[ty_local - i + FILTER_RADIUS][tx_local - j + FILTER_RADIUS] = 0.0f;
            }
        }
    }

    // top-right
    if (tx_local == INNER_TILE_DIM - 1 && ty_local == 0) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            for (int j = 1; j <= FILTER_RADIUS; ++j) {
                if (row - i >= 0 && col + j < width)
                    N_s[ty_local - i + FILTER_RADIUS][tx_local + j + FILTER_RADIUS] = N[(row - i) * width + (col + j)];
                else
                    N_s[ty_local - i + FILTER_RADIUS][tx_local + j + FILTER_RADIUS] = 0.0f;
            }
        }
    }

    // bottom-left
    if (tx_local == 0 && ty_local == INNER_TILE_DIM - 1) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            for (int j = 1; j <= FILTER_RADIUS; ++j) {
                if (row + i < height && col - j >= 0)
                    N_s[ty_local + i + FILTER_RADIUS][tx_local - j + FILTER_RADIUS] = N[(row + i) * width + (col - j)];
                else
                    N_s[ty_local + i + FILTER_RADIUS][tx_local - j + FILTER_RADIUS] = 0.0f;
            }
        }
    }

    // bottom-right
    if (tx_local == INNER_TILE_DIM - 1 && ty_local == INNER_TILE_DIM - 1) {
        for (int i = 1; i <= FILTER_RADIUS; ++i) {
            for (int j = 1; j <= FILTER_RADIUS; ++j) {
                if (row + i < height && col + j < width)
                    N_s[ty_local + i + FILTER_RADIUS][tx_local + j + FILTER_RADIUS] = N[(row + i) * width + (col + j)];
                else
                    N_s[ty_local + i + FILTER_RADIUS][tx_local + j + FILTER_RADIUS] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Compute outputs for inner threads
    if (tx_local >= 0 && tx_local < INNER_TILE_DIM && ty_local >= 0 && ty_local < INNER_TILE_DIM) {
        int out_col = bx * INNER_TILE_DIM + tx_local;
        int out_row = by * INNER_TILE_DIM + ty_local;

        if (out_row < height && out_col < width) {
            float acc = 0.0f;
            #pragma unroll
            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
                #pragma unroll
                for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
                    float w = F[(i + FILTER_RADIUS) * (2*FILTER_RADIUS+1) + (j + FILTER_RADIUS)];
                    acc += w * N_s[ty_local + i + FILTER_RADIUS][tx_local + j + FILTER_RADIUS];
                }
            }
            P[out_row * width + out_col] = acc;
        }
    }
}
            

// Variant 3: strided-load of the full OUTER tile using INNER threads
// - Threads per block are intended to be INNER_TILE_DIM x INNER_TILE_DIM
// - All threads cooperatively load the entire OUTER_TILE_DIM x OUTER_TILE_DIM tile
//   via 2D strided loops, then each thread computes one inner output
__global__ void convolution_var_3(
    const float* __restrict__ N, float* __restrict__ P,
    int width, int height)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // use same mapping as var_1 so grid (which advances by INNER_TILE_DIM)
    // still matches output tiling even when blockDim = OUTER_TILE_DIM
    int col = blockIdx.x * INNER_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * INNER_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[OUTER_TILE_DIM][OUTER_TILE_DIM];

    int tile_row0 = by * INNER_TILE_DIM - FILTER_RADIUS;
    int tile_col0 = bx * INNER_TILE_DIM - FILTER_RADIUS;

    // stride by the runtime blockDim so this works when blockDim == OUTER_TILE_DIM
    for(int j = ty ; j < OUTER_TILE_DIM ; j += blockDim.y) {
        for(int i = tx ; i < OUTER_TILE_DIM ; i += blockDim.x) {
            int load_x = tile_col0 + i;
            int load_y = tile_row0 + j;

            if (load_x >= 0 && load_x < width && load_y >=0 && load_y < height) {
                N_s[j][i] = N[load_y*width + load_x];
            }
            else {
                N_s[j][i] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Only inner threads perform the convolution computation, mirroring var_1
    int tCol = threadIdx.x - FILTER_RADIUS;
    int tRow = threadIdx.y - FILTER_RADIUS;

    if (tCol >= 0 && tCol < INNER_TILE_DIM && tRow >= 0 && tRow < INNER_TILE_DIM) {
        float val = 0.0f;
        #pragma unroll
        for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy) {
            #pragma unroll
            for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx) {
                float w = F[(dy + FILTER_RADIUS) * (2*FILTER_RADIUS+1) + (dx + FILTER_RADIUS)];
                // Access within the shared-memory tile is guaranteed in-bounds for inner threads
                val += w * N_s[threadIdx.y + dy][threadIdx.x + dx];
            }
        }

        int out_col = blockIdx.x * INNER_TILE_DIM + tCol;
        int out_row = blockIdx.y * INNER_TILE_DIM + tRow;
        if (out_row < height && out_col < width) {
            P[out_row * width + out_col] = val;
        }
    }
}


int main(int argc, char** argv) {
    const int width = 1024, height = 1024;
    const int n = width * height;
    const size_t bytes = n * sizeof(float);

    std::vector<float> hF((2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1), 1.0f);
    for (auto &x : hF) x /= float((2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1));

    std::vector<float> hN(n, 1.0f);
    std::vector<float> hP(n, 0.0f);

    float *dN = nullptr, *dP = nullptr;
    cudaMalloc(&dN, bytes);
    cudaMalloc(&dP, bytes);

    cudaMemcpy(dN, hN.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, hF.data(),
        sizeof(float) * (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1));

    dim3 grid((width  + INNER_TILE_DIM - 1) / INNER_TILE_DIM,
              (height + INNER_TILE_DIM - 1) / INNER_TILE_DIM);
    dim3 block(OUTER_TILE_DIM, OUTER_TILE_DIM);

    // Parse command line argument
    int variant = 1;
    if (argc > 1) {
        if (strcmp(argv[1], "var_1") == 0) variant = 1;
        else if (strcmp(argv[1], "var_2") == 0) variant = 2;
        else if (strcmp(argv[1], "var_3") == 0) variant = 3;
    }

    switch(variant) {
        case 1:
            convolution_var_1<<<grid, block>>>(dN, dP, width, height);
            cudaDeviceSynchronize();
            cudaMemcpy(hP.data(), dP, bytes, cudaMemcpyDeviceToHost);
            printf("var_1: P[0,0] = %f\n", hP[0]);
            break;
        case 2:
            convolution_var_2<<<grid, block>>>(dN, dP, width, height);
            cudaDeviceSynchronize();
            cudaMemcpy(hP.data(), dP, bytes, cudaMemcpyDeviceToHost);
            printf("var_2: P[0,0] = %f\n", hP[0]);
            break;
        case 3:
            convolution_var_3<<<grid, block>>>(dN, dP, width, height);
            cudaDeviceSynchronize();
            cudaMemcpy(hP.data(), dP, bytes, cudaMemcpyDeviceToHost);
            printf("var_3: P[0,0] = %f\n", hP[0]);
            break;
    }

    cudaFree(dN);
    cudaFree(dP);
    return 0;
}