#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function must be outside the host function
__global__ void matmulkernel(float* Md, float* Nd, float* Pd, int Width)
{
    // 2D thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // P value stores the Pd element that is computed by the thread
    float Pvalue = 0;

    for (int k = 0; k < Width; ++k)
    {
        float Mdelement = Md[ty * Width + k];
        float Ndelement = Nd[k * Width + tx];
        Pvalue += Mdelement * Ndelement;
    }

    // Write the matrix to device memory; each thread writes one element
    Pd[ty * Width + tx] = Pvalue;
}

void matmul(float* M, float* N, float* P, int Width)
{
    int size = Width * Width * sizeof(float);
    float *Md, *Nd, *Pd; // Corrected pointer declarations

    // 1. Allocate device memory for Md, Nd, Pd
    cudaMalloc((void**)&Md, size);
    cudaMalloc((void**)&Nd, size);
    cudaMalloc((void**)&Pd, size);

    // 2. Copy M and N to allocated device memory locations
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    // 3. Setup execution configuration
    dim3 dimBlock(Width, Width);
    dim3 dimGrid(1, 1);

    // 4. Launch the device computational threads
    matmulkernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);

    // 5. Copy P from the device memory
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    // 6. Free device matrices from device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}