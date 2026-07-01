// P1 Week 5 - many small kernels as an Nsight Systems sample.
// Build & run:
//   nvcc -std=c++17 13_many_small_kernels.cu -o many_small && ./many_small
//
// Suggested profiling:
//   nsys profile -o many_small_report ./many_small

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void tinyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    const int n = 256;
    const std::size_t bytes = n * sizeof(float);

    float* dData = nullptr;
    CHECK_CUDA(cudaMalloc(&dData, bytes));
    CHECK_CUDA(cudaMemset(dData, 0, bytes));

    for (int iter = 0; iter < 500; ++iter) {
        tinyKernel<<<1, 256>>>(dData, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "launched 500 tiny kernels\n";

    CHECK_CUDA(cudaFree(dData));
    return EXIT_SUCCESS;
}
