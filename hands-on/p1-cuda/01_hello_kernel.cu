// P1 Week 1 - the first CUDA kernel.
// Build & run:
//   nvcc -std=c++17 01_hello_kernel.cu -o hello && ./hello
//
// Every GPU thread runs the same kernel body once.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf(
        "Hello from block %d thread %d\n",
        blockIdx.x,
        threadIdx.x
    );
}

int main() {
    helloKernel<<<2, 4>>>();

    // Wait for the GPU to finish so the printf output is flushed before exit.
    cudaDeviceSynchronize();
    return 0;
}
