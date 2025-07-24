# Introduction to CUDA

## Key Concepts
- **Host** vs **Device**
  - **Host**: The CPU and its memory (RAM). Code that runs on the host is standard C/C++ code.
  - **Device**: The GPU and its memory (VRAM) — the graphics card's processors and dedicated memory. Code that runs on the device consists of CUDA kernels (special functions).

## Data Parallelism
- Data parallelism refers to the property of a program whereby many arithmetic operations can be safely performed on data structures simultaneously.
- A classic example of data parallelism is matrix-matrix multiplication, where each element of the result matrix can be computed independently and in parallel.

### Demonstration: Matrix-Matrix Multiplication

Given two matrices \(A\) and \(B\), their product \(C = A \times B\) is defined as:

\[
C_{i,j} = \sum_{k=1}^{N} A_{i,k} \times B_{k,j}
\]

Where:
- \(A\) is of size \(M \times N\)
- \(B\) is of size \(N \times P\)
- \(C\) is of size \(M \times P\)

#### Pseudocode

```c
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < P; ++j) {
        C[i][j] = 0;
        for (int k = 0; k < N; ++k) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

**Parallelism:**  
Each computation of \(C_{i,j}\) is independent and can be performed in parallel, making this operation ideal for GPU acceleration using CUDA.

**Cuda Program Structure**
A CUDA program consists of one or more phases that are executed on either the host (CPU) or a device such as a GPU.
A CUDA program is a unified source code encompassing both host and device code.
The NVIDIA C compiler (nvcc) seperates the two during compilation process.
The host code is straight ANSI C code; further compiled with the host's standard C compilers and runs as an ordinary CPU process.
The device code is written using ANSI C extended with keywords for labelling data-parallel functions, called `kernels` and their associated data structures.
The device code is typically compiled by the `nvcc` and executed on a GPU device.
The kernel functions typically generatea large number of threads to exploit data parallelism.
In the matrix multiplication example,the matrix multiplication computation can be implemented as a kernel where each thread is used to compute one element of the output matrix `C`.
In that example,the number of threads used by the kernel is a function of the matrix dimension.

### Steps in the Execution of a CUDA Program

1. **Allocate memory on the device (GPU):**
   - Use CUDA API functions (e.g., `cudaMalloc`) to allocate memory on the GPU for input and output data.

2. **Transfer data from host to device:**
   - Copy input data from the host (CPU) memory to the device (GPU) memory using functions like `cudaMemcpy`.

3. **Configure and launch the kernel:**
   - Specify the number of threads and blocks, then launch the kernel function on the device to perform parallel computation.

4. **Transfer results from device to host:**
   - Copy the computed results from device memory back to host memory using `cudaMemcpy`.

5. **Free device memory:**
   - Release the allocated memory on the device using `cudaFree` to avoid memory leaks.

6. **Process results on the host (if needed):**
   - Perform any further processing or output on the CPU as required.


**Device Memories and Data Transfer**

CUDA provides a memory model that distinguishes between host (CPU) memory and device (GPU) memory. Data must be explicitly transferred between these two memory spaces using CUDA API functions.

- **Global Memory:** The main memory space on the device (GPU). It is accessible by all threads but has higher latency compared to other memory types.
- **Shared Memory:** Fast, on-chip memory shared among threads within the same block.
- **Constant and Texture Memory:** Specialized read-only memory spaces optimized for specific access patterns.

**Memory Management Functions:**

- `cudaMalloc(void **devPtr, size_t size)`: Allocates `size` bytes of linear memory on the device and returns a pointer to the allocated memory in `*devPtr`.
  - The first parameter is the address of a pointer variable that will point to the allocated device memory after allocation. It should be cast to `(void **)` because `cudaMalloc` is a generic function.
  - The second parameter specifies the size of the allocation in bytes, similar to the `malloc()` function in C.

- `cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)`: Copies data between host and device memory.
  - `dst` and `src` are the destination and source pointers.
  - `count` is the number of bytes to copy.
  - `kind` specifies the direction of the copy (e.g., `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`).

**Kernel Functions and Threading:**

- **Kernel Functions:**  
  - A kernel is a function written in C/C++ with CUDA extensions that runs on the device (GPU).
  - Kernels are declared with the `__global__` qualifier.
  - When a kernel is launched, it is executed by many threads in parallel on the GPU.
  - Example declaration:
    ```c
    __global__ void myKernel(float *A, float *B, float *C, int N) {
        // Kernel code here
    }
    ```
  - Kernels are launched from the host using the special triple angle bracket syntax:
    ```c
    myKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    ```

- **Thread Hierarchy:**
  - Threads are organized into a hierarchy:
    - **Grid:** The entire collection of threads launched for a kernel.
    - **Block:** A group of threads that can cooperate via shared memory and can be synchronized.
    - **Thread:** The individual execution unit.
  - Each thread and block has unique IDs accessible via built-in variables:
    - `blockIdx.x`, `blockIdx.y`, ... : Block index within the grid.
    - `threadIdx.x`, `threadIdx.y`, ... : Thread index within the block.
    - `blockDim.x`, `blockDim.y`, ... : Dimensions of a block.
    - `gridDim.x`, `gridDim.y`, ... : Dimensions of the grid.

- **Example: Using Thread and Block Indices**
    ```c
    __global__ void addVectors(float *A, float *B, float *C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] + B[idx];
        }
    }
    // Launch with enough blocks and threads to cover N elements:
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    ```

- **Synchronization:**
  - Threads within a block can synchronize using `__syncthreads()`.
  - There is no built-in way to synchronize threads across different blocks.

- **Best Practices:**
  - Choose block and grid sizes to maximize occupancy and performance.
  - Use shared memory for data that needs to be shared among threads in a block.
  - Avoid divergent branches within a warp (group of 32 threads).

**Summary:**  
- Kernels are special functions executed in parallel by many threads on the GPU.
- Threads are organized into blocks and grids, and each has unique indices for data access.
- Proper configuration of threads and blocks is essential for efficient parallel computation.

**Example: Allocating and Transferring Data**

```c
float *h_A;           // Host pointer
float *d_A;           // Device pointer
size_t size = N * sizeof(float);

// Allocate memory on the host
h_A = (float*)malloc(size);

// Allocate memory on the device
cudaMalloc((void**)&d_A, size);

// Copy data from host to device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

// ... perform computation on device ...

// Copy results back from device to host
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_A);

// Free host memory
free(h_A);
```

**Summary:**  
- Always allocate memory on the device using `cudaMalloc` before launching kernels.
- Use `cudaMemcpy` to transfer data between host and device.
- Free device memory with `cudaFree` to prevent memory leaks.

Understanding and managing device memory is crucial for efficient CUDA programming and for ensuring correct data flow between the CPU and GPU.