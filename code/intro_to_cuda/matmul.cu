#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float sum = 0.0f;
        for (int k = 0; k < Width; ++k) {
            float a = A[row * Width + k];
            float b = B[k * Width + col];
            sum += a * b;
        }
        C[row * Width + col] = sum;
    }
}

// Host function to call the CUDA kernel
void matmul(float *A, float *B, float *C, int Width) {
    float *d_A, *d_B, *d_C;
    size_t size = Width * Width * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, Width);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
