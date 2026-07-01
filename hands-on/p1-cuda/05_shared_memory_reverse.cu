// P1 Week 2 - use shared memory inside one block.
// Build & run:
//   nvcc -std=c++17 05_shared_memory_reverse.cu -o reverse && ./reverse
//
// Each thread loads one value into shared memory.
// After __syncthreads(), each thread reads a partner position from the same tile.

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

__global__ void reverseWithinBlock(const int* input, int* output, int n) {
    __shared__ int tile[8];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    
    if (globalIdx < n) {
        tile[localIdx] = input[globalIdx];
    }

    __syncthreads();

    int reversedLocalIdx = blockDim.x - 1 - localIdx;
    if (globalIdx < n) {
        output[globalIdx] = tile[reversedLocalIdx];
    }
}

int main() {
    const int n = 16;
    const std::size_t bytes = n * sizeof(int);

    std::vector<int> hInput(n);
    std::vector<int> hOutput(n, 0);
    for (int i = 0; i < n; ++i) {
        hInput[i] = i;
    }

    int* dInput = nullptr;
    int* dOutput = nullptr;
    cudaMalloc(&dInput, bytes);
    cudaMalloc(&dOutput, bytes);

    cudaMemcpy(dInput, hInput.data(), bytes, cudaMemcpyHostToDevice);

    reverseWithinBlock<<<2, 8>>>(dInput, dOutput, n);
    cudaMemcpy(hOutput.data(), dOutput, bytes, cudaMemcpyDeviceToHost);

    std::cout << "input : ";
    for (int x : hInput) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::cout << "output: ";
    for (int x : hOutput) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    cudaFree(dInput);
    cudaFree(dOutput);
    return 0;
}
