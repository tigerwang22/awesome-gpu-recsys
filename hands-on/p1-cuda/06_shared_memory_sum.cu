// P1 Week 2 - block-level reduction with shared memory.
// Build & run:
//   nvcc -std=c++17 06_shared_memory_sum.cu -o block_sum && ./block_sum
//
// This example uses one block of 8 threads to sum 8 values:
//   1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

__global__ void blockSum(const float* input, float* output) {
    __shared__ float tile[8];

    int tid = threadIdx.x;
    tile[tid] = input[tid];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            tile[tid] += tile[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = tile[0];
    }
}

int main() {
    const int n = 8;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> hInput = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> hOutput(1, 0.0f);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    cudaMalloc(&dInput, bytes);
    cudaMalloc(&dOutput, sizeof(float));

    cudaMemcpy(dInput, hInput.data(), bytes, cudaMemcpyHostToDevice);
    blockSum<<<1, n>>>(dInput, dOutput);
    cudaMemcpy(hOutput.data(), dOutput, sizeof(float), cudaMemcpyDeviceToHost);

    const float expected = 36.0f;
    if (std::fabs(hOutput[0] - expected) < 1e-6f) {
        std::cout << "blockSum passed: " << hOutput[0] << "\n";
    } else {
        std::cout << "blockSum failed: got " << hOutput[0]
                  << ", expected " << expected << "\n";
    }

    cudaFree(dInput);
    cudaFree(dOutput);
    return EXIT_SUCCESS;
}
