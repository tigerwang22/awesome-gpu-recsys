// P1 Week 6 - sweep block sizes for the same kernel.
// Build & run:
//   nvcc -std=c++17 w6_01_block_size_sweep.cu -o block_size_sweep && ./block_size_sweep

#include <cstdlib>
#include <iostream>
#include <vector>

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

__global__ void saxpyGridStride(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

float runConfig(int n, float* dX, float* dY, int threadsPerBlock) {
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        saxpyGridStride<<<blocks, threadsPerBlock>>>(n, 2.0f, dX, dY);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / 100.0f;
}

int main() {
    const int n = 1 << 24;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> hX(n, 1.0f);
    std::vector<float> hY(n, 2.0f);

    float* dX = nullptr;
    float* dY = nullptr;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dY, bytes));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));

    const int candidates[] = {64, 128, 256, 512};
    for (int threadsPerBlock : candidates) {
        CHECK_CUDA(cudaMemcpy(dY, hY.data(), bytes, cudaMemcpyHostToDevice));
        float avgMs = runConfig(n, dX, dY, threadsPerBlock);
        std::cout << "threadsPerBlock=" << threadsPerBlock
                  << ", average kernel time=" << avgMs << " ms\n";
    }

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    return EXIT_SUCCESS;
}
