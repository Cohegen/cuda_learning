#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmulkernel(float *Md, float *Nd, float *Pd, int Width)
{
    // calculate the row index of the Pd element and M
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // calculate the column index of Pd and N
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Pvalue = 0;
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < Width; k++)
    {
        Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];
    }
    Pd[Row * Width + Col] = Pvalue;
}

int main()
{
    int Width = 1024; // Example value, set as needed
    size_t size = Width * Width * sizeof(float);

    // Allocate host memory
    float *M = (float *)malloc(size);
    float *N = (float *)malloc(size);
    float *P = (float *)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < Width * Width; i++) {
        M[i] = 1.0f; // or any value you want
        N[i] = 2.0f;
    }

    // Allocate device memory
    float *Md, *Nd, *Pd;
    cudaMalloc((void **)&Md, size);
    cudaMalloc((void **)&Nd, size);
    cudaMalloc((void **)&Pd, size);

    // Copy host memory to device
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // launching the device computation threads
    matmulkernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);

    // Copy result back to host
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    // (Optional) Print part of the result for verification
    printf("P[0]=%f\n", P[0]);

    // Free device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);

    // Free host memory
    free(M);
    free(N);
    free(P);

    return 0;
}